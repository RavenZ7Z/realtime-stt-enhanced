import sys, os
import re
import time
from pathlib import Path

import sherpa_onnx
import onnxruntime
import numpy as np
import sounddevice as sd
import wave

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton,
    QPlainTextEdit, QFileDialog, QLabel, QMessageBox
)
from PySide6.QtCore import QThread, Signal, Qt, QUrl, Slot
from PySide6.QtGui import QIcon, QCloseEvent, QDesktopServices

ROOT_DIR = Path(os.getcwd()).as_posix()
MODEL_DIR = f'{ROOT_DIR}/onnx'
OUT_DIR = f'{ROOT_DIR}/output'
Path(MODEL_DIR).mkdir(exist_ok=True)
Path(OUT_DIR).mkdir(exist_ok=True)
CTC_MODEL_FILE = f"{MODEL_DIR}/ctc.model.onnx"
PAR_ENCODER = f"{MODEL_DIR}/encoder.onnx"
PAR_DECODER = f"{MODEL_DIR}/decoder.onnx"
PAR_TOKENS = f"{MODEL_DIR}/tokens.txt"

if sys.platform == 'win32':
    os.environ['PATH'] = f'{ROOT_DIR};{ROOT_DIR}/ffmpeg;' + os.environ['PATH']


class OnnxModel:
    def __init__(self):
        session_opts = onnxruntime.SessionOptions()
        session_opts.log_severity_level = 3  # error level
        self.sess = onnxruntime.InferenceSession(CTC_MODEL_FILE, session_opts)

        self._init_punct()
        self._init_tokens()

    def _init_punct(self):
        meta = self.sess.get_modelmeta().custom_metadata_map
        punct = meta["punctuations"].split("|")
        self.id2punct = punct
        self.punct2id = {p: i for i, p in enumerate(punct)}

        self.dot = self.punct2id["。"]
        self.comma = self.punct2id["，"]
        self.pause = self.punct2id["、"]
        self.quest = self.punct2id["？"]
        self.underscore = self.punct2id["_"]

    def _init_tokens(self):
        meta = self.sess.get_modelmeta().custom_metadata_map
        tokens = meta["tokens"].split("|")
        self.id2token = tokens
        self.token2id = {t: i for i, t in enumerate(tokens)}

        unk = meta["unk_symbol"]
        assert unk in self.token2id, unk
        self.unk_id = self.token2id[unk]

    def __call__(self, text: str) -> str:
        word_list = text.split()

        words = []
        for w in word_list:
            s = ""
            for c in w:
                if len(c.encode()) > 1:
                    if s == "":
                        s = c
                    elif len(s[-1].encode()) > 1:
                        s += c
                    else:
                        words.append(s)
                        s = c
                else:
                    if s == "":
                        s = c
                    elif len(s[-1].encode()) > 1:
                        words.append(s)
                        s = c
                    else:
                        s += c
            if s:
                words.append(s)

        ids = []
        for w in words:
            if len(w[0].encode()) > 1:
                for c in w:
                    ids.append(self.token2id.get(c, self.unk_id))
            else:
                ids.append(self.token2id.get(w, self.unk_id))

        segment_size = 30
        num_segments = (len(ids) + segment_size - 1) // segment_size

        punctuations = []
        max_len = 200
        last = -1

        for i in range(num_segments):
            this_start = i * segment_size
            this_end = min(this_start + segment_size, len(ids))
            if last != -1:
                this_start = last

            inputs = ids[this_start:this_end]

            out = self.sess.run(
                [self.sess.get_outputs()[0].name],
                {
                    self.sess.get_inputs()[0].name: np.array(inputs, dtype=np.int32).reshape(1, -1),
                    self.sess.get_inputs()[1].name: np.array([len(inputs)], dtype=np.int32),
                },
            )[0]
            out = out[0].argmax(axis=-1).tolist()

            dot_index = -1
            comma_index = -1

            for k in range(len(out) - 1, 1, -1):
                if out[k] in (self.dot, self.quest):
                    dot_index = k
                    break
                if comma_index == -1 and out[k] == self.comma:
                    comma_index = k

            if dot_index == -1 and len(inputs) >= max_len and comma_index != -1:
                dot_index = comma_index
                out[dot_index] = self.dot

            if dot_index == -1:
                if last == -1:
                    last = this_start
                if i == num_segments - 1:
                    dot_index = len(inputs) - 1
            else:
                last = this_start + dot_index + 1

            if dot_index != -1:
                punctuations += out[: dot_index + 1]

        ans = []
        for i, p in enumerate(punctuations):
            t = self.id2token[ids[i]]
            if ans and len(ans[-1][0].encode()) == 1 and len(t[0].encode()) == 1:
                ans.append(" ")
            ans.append(t)
            if p != self.underscore:
                ans.append(self.id2punct[p])

        return "".join(ans)


def split_sentences(text: str) -> list[str]:
    """将带标点的文本按句末标点切分成多句（保留标点）。"""
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if not text:
        return []
    parts = re.split(r'(?<=[。！？!?；;])\s*', text)
    return [p.strip() for p in parts if p.strip()]


def fmt_ts(seconds: float) -> str:
    """Format seconds into mm:ss.mmm or hh:mm:ss.mmm if >= 1 hour."""
    if seconds < 0:
        seconds = 0.0
    ms_total = int(round(seconds * 1000.0))
    hh = ms_total // 3600000
    mm = (ms_total % 3600000) // 60000
    ss = (ms_total % 60000) // 1000
    ms = ms_total % 1000
    if hh > 0:
        return f"{hh:02d}:{mm:02d}:{ss:02d}.{ms:03d}"
    return f"{mm:02d}:{ss:02d}.{ms:03d}"


def create_recognizer():
    recognizer = sherpa_onnx.OnlineRecognizer.from_paraformer(
        tokens=PAR_TOKENS,
        encoder=PAR_ENCODER,
        decoder=PAR_DECODER,
        num_threads=2,
        sample_rate=16000,
        feature_dim=80,
        enable_endpoint_detection=True,
        rule1_min_trailing_silence=2.4,
        rule2_min_trailing_silence=1.2,
        rule3_min_utterance_length=20,
    )
    return recognizer


class Worker(QThread):
    new_word = Signal(str)          # realtime_text 用
    new_segment = Signal(object)    # dict: {"start": float, "end": float, "text": str}
    ready = Signal()
    session_started = Signal(str)   # txt 路径（让 GUI 同步保存）

    def __init__(self, device_idx, timestamp_str: str, parent=None):
        super().__init__(parent)
        self.device_idx = device_idx
        self.timestamp_str = timestamp_str

        self.running = False
        self.paused = False

        self.sample_rate = 48000
        self.samples_per_read = int(0.1 * self.sample_rate)

    @Slot(bool)
    def set_paused(self, v: bool):
        self.paused = v

    def run(self):
        devices = sd.query_devices()
        if len(devices) == 0:
            return

        print(f'使用麦克风: {devices[self.device_idx]["name"]}')
        punct_model = OnnxModel()
        recognizer = create_recognizer()
        stream = recognizer.create_stream()

        mic_stream = sd.InputStream(
            device=self.device_idx,
            channels=1,
            dtype="float32",
            samplerate=self.sample_rate
        )
        mic_stream.start()

        # wav 仍由 Worker 写（不会因暂停产生新片段）
        wav_path = f"{OUT_DIR}/{self.timestamp_str}.wav"
        wav_file = wave.open(wav_path, 'wb')
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(self.sample_rate)

        # txt 由 GUI 按“当前显示状态”覆盖写，保证切换时间戳后也同步
        txt_path = f"{OUT_DIR}/{self.timestamp_str}.txt"
        self.session_started.emit(txt_path)

        self.ready.emit()
        self.running = True

        last_result = ""

        total_samples = 0
        segment_start_samples = 0

        while self.running:
            if self.paused:
                time.sleep(0.05)
                continue

            samples, _ = mic_stream.read(self.samples_per_read)
            samples_int16 = (samples * 32767).astype(np.int16)
            total_samples += len(samples)  # (frames,1)
            wav_file.writeframes(samples_int16.tobytes())

            samples = samples.reshape(-1)
            stream.accept_waveform(self.sample_rate, samples)
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)

            is_endpoint = recognizer.is_endpoint(stream)
            result = recognizer.get_result(stream)

            if result != last_result:
                self.new_word.emit(result)
                last_result = result

            if is_endpoint:
                if result:
                    punctuated = punct_model(result)
                    lines = split_sentences(punctuated)

                    if not lines:
                        segment_start_samples = total_samples
                        recognizer.reset(stream)
                        continue

                    seg_start = segment_start_samples / self.sample_rate
                    seg_end = total_samples / self.sample_rate

                    for line in lines:
                        self.new_segment.emit({"start": seg_start, "end": seg_end, "text": line})

                    segment_start_samples = total_samples

                recognizer.reset(stream)

        mic_stream.stop()
        wav_file.close()


class RealTimeWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('实时语音转文字 - 支持中文和英文语音 - pyVideoTrans.com')
        self.layout = QVBoxLayout(self)
        self.setWindowIcon(QIcon(f"{ROOT_DIR}/data/icon.ico"))

        # 会话数据：结构化段落（用于切换时间戳时全量刷新）
        self.segments = []  # list[{"start":float,"end":float,"text":str}]
        self.session_txt_path = None

        # 顶部：麦克风选择 + 启动 + 暂停 + 时间戳
        self.mic_layout = QHBoxLayout()
        self.combo = QComboBox()
        self.populate_mics()
        self.mic_layout.addWidget(self.combo)

        self.start_button = QPushButton('启动实时语音转文字')
        self.start_button.setCursor(Qt.PointingHandCursor)
        self.start_button.setMinimumHeight(30)
        self.start_button.setMinimumWidth(150)
        self.start_button.clicked.connect(self.toggle_transcription)
        self.mic_layout.addWidget(self.start_button)

        self.pause_button = QPushButton('暂停')
        self.pause_button.setCursor(Qt.PointingHandCursor)
        self.pause_button.setMinimumHeight(30)
        self.pause_button.setMinimumWidth(90)
        self.pause_button.setEnabled(False)
        self.pause_button.clicked.connect(self.toggle_pause)
        self.mic_layout.addWidget(self.pause_button)

        # 时间戳按钮（可切换，默认开启）
        self.timestamp_btn = QPushButton("时间戳：开")
        self.timestamp_btn.setCheckable(True)
        self.timestamp_btn.setChecked(True)
        self.timestamp_btn.setCursor(Qt.PointingHandCursor)
        self.timestamp_btn.setMinimumHeight(30)
        self.timestamp_btn.setMinimumWidth(110)
        self.timestamp_btn.toggled.connect(self.on_toggle_timestamp)
        self._update_timestamp_btn_style(True)
        self.mic_layout.addWidget(self.timestamp_btn)

        self.layout.addLayout(self.mic_layout)

        # Real-time text（不带时间戳）
        self.realtime_text = QPlainTextEdit()
        self.realtime_text.setReadOnly(True)
        self.realtime_text.setStyleSheet("background: transparent; border: none;font-size:14px")
        self.realtime_text.setMaximumHeight(80)
        self.layout.addWidget(self.realtime_text)

        # 下方最终段落（可切换显示时间戳）
        self.textedit = QPlainTextEdit()
        self.textedit.setReadOnly(True)
        self.textedit.setMinimumHeight(400)
        self.textedit.setStyleSheet("color:#ffffff")
        self.layout.addWidget(self.textedit)

        # Buttons layout
        self.button_layout = QHBoxLayout()
        self.export_button = QPushButton('导出为TXT')
        self.export_button.clicked.connect(self.export_txt)
        self.export_button.setCursor(Qt.PointingHandCursor)
        self.export_button.setMinimumHeight(35)
        self.button_layout.addWidget(self.export_button)

        self.copy_button = QPushButton('复制')
        self.copy_button.setMinimumHeight(35)
        self.copy_button.setCursor(Qt.PointingHandCursor)
        self.copy_button.clicked.connect(self.copy_textedit)
        self.button_layout.addWidget(self.copy_button)

        self.clear_button = QPushButton('清空')
        self.clear_button.setMinimumHeight(35)
        self.clear_button.setCursor(Qt.PointingHandCursor)
        self.clear_button.clicked.connect(self.clear_textedit)
        self.button_layout.addWidget(self.clear_button)

        self.layout.addLayout(self.button_layout)

        self.btn_opendir = QPushButton(f"录音文件保存到: {OUT_DIR}")
        self.btn_opendir.setStyleSheet("background-color:transparent;border:0;")
        self.btn_opendir.clicked.connect(self.open_dir)
        self.layout.addWidget(self.btn_opendir)

        self.worker = None
        self.transcribing = False
        self.paused = False

    def check_model_exist(self):
        if not Path(PAR_ENCODER).exists() or not Path(CTC_MODEL_FILE).exists() or not Path(PAR_DECODER).exists():
            QMessageBox.information(
                self, '缺少实时语音转文字所需模型，请去下载',
                f'模型下载地址已复制到剪贴板内，请到浏览器地址栏中粘贴下载\n\n为减小软件包体积，默认未内置模型，下载解压后，将其内的4个文件放到 {MODEL_DIR}  文件夹内'
            )
            QApplication.clipboard().setText('https://github.com/jianchang512/stt/releases/download/0.0/realtimestt-models.7z')
            return False
        return True

    def open_dir(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(OUT_DIR))

    def populate_mics(self):
        devices = sd.query_devices()
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        if not input_devices:
            print("未找到任何可用麦克风")
            sys.exit(0)

        default_idx = sd.default.device[0]
        default_item = 0
        for i, d in enumerate(input_devices):
            self.combo.addItem(d['name'], d['index'])
            if d['index'] == default_idx:
                default_item = i
        self.combo.setCurrentIndex(default_item)

    def toggle_transcription(self):
        if self.check_model_exist() is not True:
            return

        if not self.transcribing:
            # start
            self.realtime_text.setPlainText('请稍等...')

            self.segments = []
            self.session_txt_path = None
            self.refresh_segments_view()

            device_idx = self.combo.currentData()
            timestamp_str = time.strftime("%Y%m%d_%H-%M-%S")

            self.worker = Worker(device_idx, timestamp_str)
            self.worker.new_word.connect(self.update_realtime)
            self.worker.new_segment.connect(self.append_segment)
            self.worker.ready.connect(self.update_realtime_ready)
            self.worker.session_started.connect(self.on_session_started)
            self.worker.start()

            self.start_button.setText('正在语音转文字中...')
            self.transcribing = True

            self.pause_button.setEnabled(True)
            self.pause_button.setText("暂停")
            self.paused = False
            self.start_button.setEnabled(True)

        else:
            # stop
            if self.worker:
                self.worker.running = False
                self.worker.wait()
                self.worker = None

            self.start_button.setText('启动实时转录')
            self.transcribing = False

            self.pause_button.setEnabled(False)
            self.pause_button.setText("暂停")
            self.paused = False
            self.start_button.setEnabled(True)

            # 注意：停止时不把 realtime_text 追加到 textedit（避免无时间戳的“半句”混进最终段落）
            self.realtime_text.clear()

    @Slot(str)
    def _update_timestamp_btn_style(self, on: bool):
        # 更明显：开=高亮，关=灰
        if on:
            self.timestamp_btn.setText("时间戳：开")
            self.timestamp_btn.setStyleSheet(
                "font-size:15px;font-weight:800;padding:6px 10px;"
                "border-radius:6px;"
            )
        else:
            self.timestamp_btn.setText("时间戳：关")
            self.timestamp_btn.setStyleSheet(
                "font-size:15px;font-weight:800;padding:6px 10px;"
                "border-radius:6px;opacity:0.8;"
            )

    def on_toggle_timestamp(self, on: bool):
        self._update_timestamp_btn_style(on)
        self.refresh_segments_view()
    def on_session_started(self, txt_path: str):
        self.session_txt_path = txt_path
        self.refresh_segments_view()

    def update_realtime(self, text):
        # 上方实时区域永远不带时间戳
        self.realtime_text.setPlainText(text)
        scrollbar = self.realtime_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def update_realtime_ready(self):
        self.realtime_text.setPlainText('请说话...')

    def render_segment(self, seg: dict) -> str:
        if self.timestamp_btn.isChecked():
            return f"[{fmt_ts(seg['start'])} - {fmt_ts(seg['end'])}] {seg['text']}"
        return seg["text"]

    def refresh_segments_view(self):
        rendered = "\n".join(self.render_segment(s) for s in self.segments)
        self.textedit.setPlainText(rendered)
        scrollbar = self.textedit.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

        # 同步覆盖写会话 txt（与 GUI 一致）
        if self.session_txt_path:
            with open(self.session_txt_path, "w", encoding="utf-8") as f:
                f.write(rendered)

    def append_segment(self, seg):
        # seg: {"start":..., "end":..., "text":...}
        self.segments.append(seg)
        self.refresh_segments_view()

    def toggle_pause(self):
        if not self.worker or not self.transcribing:
            return

        self.paused = not self.paused
        self.worker.set_paused(self.paused)

        if self.paused:
            self.pause_button.setText("继续")
            self.start_button.setEnabled(False)  # 你要求：暂停时灰掉启动按钮
            self.realtime_text.setPlainText("已暂停")
        else:
            self.pause_button.setText("暂停")
            self.start_button.setEnabled(True)
            self.realtime_text.setPlainText("请说话...")

    def export_txt(self):
        text = self.textedit.toPlainText().strip()
        if not text:
            return
        file_name, _ = QFileDialog.getSaveFileName(self, "Save TXT", "", "Text files (*.txt)")
        if file_name:
            if not file_name.endswith(".txt"):
                file_name += ".txt"
            with open(file_name, 'w', encoding='utf-8') as f:
                f.write(text)

    def copy_textedit(self):
        text = self.textedit.toPlainText()
        QApplication.clipboard().setText(text)

    def clear_textedit(self):
        self.segments = []
        self.refresh_segments_view()

    def closeEvent(self, event: QCloseEvent):
        if self.transcribing:
            self.toggle_transcription()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    with open(f'{ROOT_DIR}/data/style.qss', 'r', encoding='utf-8') as f:
        app.setStyleSheet(f.read())
    window = RealTimeWindow()
    window.show()
    sys.exit(app.exec())