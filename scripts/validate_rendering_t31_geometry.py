import glob,os,sys,warnings
warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0,"scripts"); sys.path.insert(0,"src")
import validate_rendering_t31 as v
from PIL import Image
from scipy.ndimage import distance_transform_edt as edt
def mask(a): return a.max(axis=2)>30
for pid,cfg in v.PARTICIPANTS.items():
    reals={os.path.basename(p):mask(v.load_real_image(p)) for p in sorted(glob.glob(os.path.join(cfg["images_dir"],cfg["pattern"])))}
    dts={n:edt(~m) for n,m in reals.items()}
    rows=[]
    for f in glob.glob(f"{v.OUT_DIR}/p{pid}_*.png"):
        r=mask(np.asarray(Image.open(f).convert("RGB").resize((640,480)),dtype=float))
        if r.sum()<50: continue
        for n,m in reals.items():
            # symmetric mean chamfer distance (px)
            c=(dts[n][r].mean()+edt(~r)[m].mean())/2
            rows.append((round(float(c),2),os.path.basename(f),n,int(r.sum()),int(m.sum())))
    rows.sort(); c=np.array([x[0] for x in rows])
    print(pid,"median chamfer",np.median(c).round(1),"best",rows[0][0])
    for x in rows[:4]: print("  ",x)
