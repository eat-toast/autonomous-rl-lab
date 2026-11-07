"""Mix the user-selected Pixabay track into the approved 11-second edit."""
from pathlib import Path
import sys, subprocess, json, hashlib
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'.tools')); sys.path.insert(0,str(ROOT))
import imageio_ffmpeg
from scripts.commit_review import validate_review, commit_review
validate_review(ROOT/'studies/MLP-01/results')
commit=commit_review(ROOT/'studies/MLP-01/results')
base=ROOT/'studies/MLP-01/exports/instagram_MLP-01_EP02'
out=base/'pixel_music_v1'; out.mkdir(exist_ok=True)
audio=ROOT/'assets/instagram/audio/pixabay-527182/reganati-chillcalming-chiptune-loop-527182.mp3'
if not audio.is_file():
    raise FileNotFoundError('Instagram audio library missing pixabay-527182; see docs/instagram_music_library.md')
video=out/'pixel_maze_with_music.mp4'
ff=imageio_ffmpeg.get_ffmpeg_exe()
filters='atrim=start=0:end=11,asetpts=PTS-STARTPTS,loudnorm=I=-20:TP=-2:LRA=7,afade=t=in:st=0:d=0.08,afade=t=out:st=10.3:d=0.7'
subprocess.run([ff,'-y','-v','error','-i',str(base/'pixel_pilot_v2/pixel_maze_pilot_v2.mp4'),'-i',str(audio),'-map','0:v:0','-map','1:a:0','-c:v','copy','-af',filters,'-c:a','aac','-b:a','192k','-t','11','-movflags','+faststart',str(video)],check=True)
result=subprocess.run([ff,'-v','error','-i',str(video),'-progress','pipe:1','-f','null','-'],check=True,capture_output=True,text=True)
(out/'decode_check.txt').write_text(result.stdout,encoding='utf-8')
record={'review_commit':commit,'track':'Chill/Calming Chiptune Loop','artist':'Reganati','url':'https://pixabay.com/music/electronic-chillcalming-chiptune-loop-527182/','license':'Pixabay Content License','license_url':'https://pixabay.com/service/license-summary/','retrieved':'2026-11-08','audio_sha256':hashlib.sha256(audio.read_bytes()).hexdigest(),'source_excerpt_seconds':[0,11],'audio_filter':filters,'video':'v2 stream copied without changes','next_selected_track':{'title':'Eternal Arcade','artist':'SlimeyFox','url':'https://pixabay.com/music/trap-eternal-arcade-487325/'}}
(out/'production.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(video)
