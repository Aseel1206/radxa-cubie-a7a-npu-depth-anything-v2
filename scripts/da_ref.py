import glob,os,numpy as np,cv2,onnxruntime as ort
R='/workspace/mp/'; D=R+'da/test'; os.makedirs(D,exist_ok=True)
fs=sorted(glob.glob(R+'images/*.jp*g')+glob.glob(R+'images/*.png'))   # any photos you like
a=ort.InferenceSession(R+'da/da_s1.onnx',providers=['CPUExecutionProvider']); b=ort.InferenceSession(R+'da/da_npu.onnx',providers=['CPUExecutionProvider'])
mean=np.array([123.675,116.28,103.53],np.float32); std=np.array([58.395,57.12,57.375],np.float32)
for k,f in enumerate(fs):
    im=cv2.imread(f); h,w=im.shape[:2]; s=max(336/w,252/h); im=cv2.resize(im,(round(w*s),round(h*s)),interpolation=cv2.INTER_AREA); h,w=im.shape[:2]; y,x=(h-252)//2,(w-336)//2; im=im[y:y+252,x:x+336]
    cv2.imwrite(f'{D}/d_{k:02d}.png',im); rgb=cv2.cvtColor(im,cv2.COLOR_BGR2RGB); rgb.tofile(f'{D}/d_{k:02d}.in.bin')
    x=((rgb.astype(np.float32)-mean)/std).transpose(2,0,1)[None]
    r=a.run(None,{'pixel_values':x})[0].ravel(); p=b.run(None,{'pixel_values':x})[0].ravel(); np.save(f'{D}/d_{k:02d}.ref.npy',r)
    if k<3: print('patched vs orig corr %.5f maxdiff %.3f (range %.1f..%.1f)'%(np.corrcoef(r,p)[0,1],abs(r-p).max(),r.min(),r.max()))
print(len(fs))
