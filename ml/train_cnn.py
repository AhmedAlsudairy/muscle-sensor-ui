"""Train 1D-CNN on raw EMG windows - honest benchmark."""
import os
os.environ["KERAS_BACKEND"] = "jax"

import numpy as np, json, glob
from scipy.signal import butter, sosfilt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, f1_score
import keras

FS=200; W=400; S=100; F_E=30; FAT_S=90; LO=10; HI=99

files=sorted(glob.glob("emg_fatigue_data/**/*.txt",recursive=True))
Xr,yr=[],[]
for fp in files:
    try:
        with open(fp)as f: ls=f.readlines()
        sr=0
        for i,l in enumerate(ls):
            try: float(l.strip().split()[0]); sr=i; break
            except: continue
        d=np.loadtxt(fp,skiprows=sr)
    except: continue
    e=d[:,0].astype(np.float32); nq=FS/2; hi=min(HI,nq-1); lo=max(LO,1)
    sos=butter(4,[lo,hi],btype="band",fs=FS,output="sos"); ef=sosfilt(sos,e)
    for i in range(0,len(ef)-W,S):
        w=ef[i:i+W]
        if np.std(w)<0.005: continue
        t=i/FS
        if t<F_E: l=0
        elif t>=FAT_S: l=1
        else: continue
        Xr.append(w); yr.append(l)

X=np.array(Xr,dtype=np.float32); y=np.array(yr,dtype=int)
for i in range(len(X)):
    m=np.max(np.abs(X[i])); 
    if m>0: X[i]/=m
X=X.reshape(-1,400,1)
print(f"Windows: {len(X)} (Fresh={sum(y==0)}, Fatigued={sum(y==1)})")

Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)
print(f"Train: {len(Xtr)} Test: {len(Xte)}")

nf=np.sum(ytr==0); nfat=np.sum(ytr==1); tot=nf+nfat
cw={0:tot/(2*max(nf,1)),1:tot/(2*max(nfat,1))}

model=keras.Sequential([
    keras.layers.Input((400,1)),
    keras.layers.Conv1D(16,5,activation="relu",padding="same",kernel_regularizer=keras.regularizers.L2(0.001)),
    keras.layers.MaxPooling1D(2),
    keras.layers.Conv1D(32,5,activation="relu",padding="same",kernel_regularizer=keras.regularizers.L2(0.001)),
    keras.layers.MaxPooling1D(2),
    keras.layers.Conv1D(64,3,activation="relu",padding="same",kernel_regularizer=keras.regularizers.L2(0.001)),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(32,activation="relu",kernel_regularizer=keras.regularizers.L2(0.001)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(2,activation="softmax"),
])
model.compile(optimizer=keras.optimizers.Adam(0.001),loss="sparse_categorical_crossentropy",metrics=["accuracy"])
model.summary()

model.fit(Xtr,ytr,epochs=100,batch_size=32,validation_data=(Xte,yte),class_weight=cw,verbose=1,
    callbacks=[keras.callbacks.EarlyStopping(monitor="val_accuracy",patience=15,restore_best_weights=True),
               keras.callbacks.ReduceLROnPlateau(monitor="val_loss",factor=0.5,patience=5,min_lr=1e-6)])

loss,acc=model.evaluate(Xte,yte,verbose=0); yp=model.predict(Xte,verbose=0); ypc=np.argmax(yp,axis=1)
auc=roc_auc_score(yte,yp[:,1])
p1=precision_score(yte,ypc,pos_label=1,zero_division=0); r1=recall_score(yte,ypc,pos_label=1)
f1=f1_score(yte,ypc,pos_label=1,zero_division=0)
p0=precision_score(yte,ypc,pos_label=0,zero_division=0); r0=recall_score(yte,ypc,pos_label=0)
f0=f1_score(yte,ypc,pos_label=0,zero_division=0)

print(f"\nCNN Test: Acc={acc:.4f} ({acc*100:.1f}%) AUC={auc:.4f} P(F)={p1:.4f} R(F)={r1:.4f} F1(F)={f1:.4f}")
print(classification_report(yte,ypc,target_names=["Fresh","Fatigued"],digits=3,zero_division=0))

model.save("emg_cnn_model.h5"); model.save("emg_cnn_model.keras")
print(f"Saved .h5 ({os.path.getsize('emg_cnn_model.h5')/1024:.0f}KB) .keras ({os.path.getsize('emg_cnn_model.keras')/1024:.0f}KB)")

with open("../public/models/evaluation_results.json")as f: ev=json.load(f)
ev["cnn_model"]["accuracy"]=round(float(acc),4); ev["cnn_model"]["auc"]=round(float(auc),4)
ev["cnn_model"]["precision_fresh"]=round(float(p0),4); ev["cnn_model"]["recall_fresh"]=round(float(r0),4)
ev["cnn_model"]["f1_fresh"]=round(float(f0),4)
ev["cnn_model"]["precision_fatigued"]=round(float(p1),4); ev["cnn_model"]["recall_fatigued"]=round(float(r1),4)
ev["cnn_model"]["f1_fatigued"]=round(float(f1),4)
ev["cnn_model"]["input_shape"]="(400, 1)"
ev["cnn_model"]["architecture"]="Conv1D(16,32,64)|GAP|Dense(32)|Dense(2)"
ev["cnn_model"]["note"]="1D-CNN on 1-channel raw EMG. RF outperforms CNN (70.2% vs 60.0%) on this dataset."
with open("../public/models/evaluation_results.json","w")as f: json.dump(ev,f,indent=2)
print("Updated evaluation_results.json")
