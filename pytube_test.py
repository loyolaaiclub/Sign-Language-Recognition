from pytube import YouTube
import certifi

url = "https://www.youtube.com/watch?v=WeAFuzYTdtU"
yt = YouTube(url, cert_path=certifi.where())
stream = yt.streams.filter(progressive=True, file_extension="mp4").order_by("resolution").desc().first()
print(stream)
