import whisper
import torch

print(torch.__version__)
print(torch.version.cuda)

# model = whisper.load_model("medium")
# result = model.transcribe(r"input\Singlish. u wan cock or flu joos _.mp3")
# print(result["text"])