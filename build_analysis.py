"""
ALY 6130 Signature Assessment - Quantitative engine
Apple acquisition of Netflix. Generates synthetic data, runs the three risk
models (R1 regulatory, R2 churn, R3 cyber), an ML churn classifier, and three
Monte Carlo simulations. Saves data to data/ and figures to figures/.
All random draws use a fixed seed for reproducibility.
"""
import numpy as np, pandas as pd, json, warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                             classification_report, accuracy_score)

ROOT="/home/claude/erm"; FIG=f"{ROOT}/figures"; RAW=f"{ROOT}/data/raw"; PROC=f"{ROOT}/data/processed"
SEED=42; rng=np.random.default_rng(SEED); np.random.seed(SEED)
NAVY="#1F4E78"; TEAL="#0E9AA7"; GOLD="#D99A2B"; RED="#C0504D"; INK="#1B2A38"
plt.rcParams.update({"font.family":"DejaVu Sans","axes.edgecolor":"#BBBBBB",
    "axes.titlecolor":NAVY,"axes.titleweight":"bold","figure.dpi":150})
def style(ax):
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    ax.grid(axis="y",color="#E2E8F0",lw=.6); ax.set_axisbelow(True)
results={}

# =====================================================================
# R2 SYNTHETIC SUBSCRIBER DATA  (drives the ML churn model)
# Assumptions: each row is one subscriber-month for the combined Apple+Netflix
# service. Features blend macro conditions (content cost ratio, competitor
# moves) with subscriber attributes (tenure, engagement, plan). Churn is drawn
# from a logistic model so the relationships are known and learnable.
# =====================================================================
N=6000
content_cost_ratio=np.clip(rng.normal(0.52,0.06,N),0.40,0.75)
competitor_moves=rng.poisson(2.0,N)
price_increase_pct=np.clip(rng.normal(3.0,2.0,N),0,12)
bundle_member=rng.binomial(1,0.40,N)
tenure_months=np.clip(rng.exponential(18,N),1,120).round(0)
engagement_hours=np.clip(rng.normal(45,20,N),0,200).round(1)
support_tickets=rng.poisson(0.5,N)
region=rng.choice(["NA","EU","APAC","LATAM"],N,p=[0.40,0.30,0.20,0.10])
plan_type=rng.choice(["Basic","Standard","Premium","AdTier"],N,p=[0.20,0.40,0.25,0.15])
region_eff=np.array([{"NA":0.0,"EU":0.1,"APAC":0.2,"LATAM":0.3}[x] for x in region])
plan_eff=np.array([{"Basic":0.2,"Standard":0.0,"Premium":-0.2,"AdTier":0.3}[x] for x in plan_type])
logit=(-1.4 + 1.2*(content_cost_ratio-0.52)*5 + 0.22*competitor_moves + 0.14*price_increase_pct
       - 0.9*bundle_member - 0.020*tenure_months - 0.012*engagement_hours + 0.35*support_tickets
       + region_eff + plan_eff)
p_churn=1/(1+np.exp(-logit))
churn=rng.binomial(1,p_churn)
df=pd.DataFrame({"content_cost_ratio":content_cost_ratio.round(3),"competitor_moves":competitor_moves,
    "price_increase_pct":price_increase_pct.round(2),"bundle_member":bundle_member,
    "tenure_months":tenure_months.astype(int),"engagement_hours":engagement_hours,
    "support_tickets":support_tickets,"region":region,"plan_type":plan_type,"churn":churn})
df.to_csv(f"{RAW}/streaming_subscribers.csv",index=False)
results["churn_base_rate"]=float(df.churn.mean())
print(f"[data] streaming_subscribers.csv  n={len(df)}  churn rate={df.churn.mean():.3f}")

# =====================================================================
# R2 ML CHURN MODEL
# =====================================================================
num=["content_cost_ratio","competitor_moves","price_increase_pct","bundle_member",
     "tenure_months","engagement_hours","support_tickets"]
cat=["region","plan_type"]
X=df[num+cat]; y=df.churn
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.30,random_state=SEED,stratify=y)
pre=ColumnTransformer([("n",StandardScaler(),num),("c",OneHotEncoder(handle_unknown="ignore"),cat)])
logreg=Pipeline([("pre",pre),("m",LogisticRegression(max_iter=1000,class_weight="balanced"))]).fit(Xtr,ytr)
rf=Pipeline([("pre",pre),("m",RandomForestClassifier(n_estimators=300,max_depth=8,random_state=SEED,class_weight="balanced"))]).fit(Xtr,ytr)
for name,mdl in [("LogReg",logreg),("RandomForest",rf)]:
    pr=mdl.predict_proba(Xte)[:,1]; auc=roc_auc_score(yte,pr); acc=accuracy_score(yte,mdl.predict(Xte))
    results[f"{name}_auc"]=round(float(auc),4); results[f"{name}_acc"]=round(float(acc),4)
    print(f"[ml] {name}: AUC={auc:.4f}  acc={acc:.4f}")
rf_proba=rf.predict_proba(Xte)[:,1]; rf_pred=rf.predict(Xte)
results["rf_report"]=classification_report(yte,rf_pred,output_dict=True)
cm=confusion_matrix(yte,rf_pred)

# ROC figure (both models)
fig,ax=plt.subplots(figsize=(6.2,4.6))
for name,mdl,c in [("Logistic regression",logreg,TEAL),("Random forest",rf,NAVY)]:
    pr=mdl.predict_proba(Xte)[:,1]; fpr,tpr,_=roc_curve(yte,pr)
    ax.plot(fpr,tpr,color=c,lw=2.2,label=f"{name} (AUC {roc_auc_score(yte,pr):.3f})")
ax.plot([0,1],[0,1],"--",color="#9AA7B2",lw=1)
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title("R2 churn model: ROC curve"); ax.legend(frameon=False,fontsize=9); style(ax)
plt.tight_layout(); plt.savefig(f"{FIG}/r2_roc_curve.png",bbox_inches="tight"); plt.close()

# Feature importance (RF) mapped back to readable names
ohe=rf.named_steps["pre"].named_transformers_["c"]
feat_names=num+list(ohe.get_feature_names_out(cat))
imp=rf.named_steps["m"].feature_importances_
fi=pd.DataFrame({"feature":feat_names,"importance":imp}).sort_values("importance",ascending=True)
fig,ax=plt.subplots(figsize=(6.4,4.8))
ax.barh(fi.feature,fi.importance,color=NAVY)
ax.set_title("R2 churn model: feature importance"); ax.set_xlabel("Importance")
for s in ["top","right"]: ax.spines[s].set_visible(False)
ax.grid(axis="x",color="#E2E8F0",lw=.6); ax.set_axisbelow(True)
plt.tight_layout(); plt.savefig(f"{FIG}/r2_feature_importance.png",bbox_inches="tight"); plt.close()
results["top_features"]=fi.tail(4).feature.tolist()[::-1]

# Confusion matrix
fig,ax=plt.subplots(figsize=(4.4,3.8))
im=ax.imshow(cm,cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j,i,cm[i,j],ha="center",va="center",
                color="white" if cm[i,j]>cm.max()/2 else INK,fontsize=13,fontweight="bold")
ax.set_xticks([0,1]); ax.set_xticklabels(["Stay","Churn"]); ax.set_yticks([0,1]); ax.set_yticklabels(["Stay","Churn"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("R2 churn model: confusion matrix")
plt.tight_layout(); plt.savefig(f"{FIG}/r2_confusion_matrix.png",bbox_inches="tight"); plt.close()

# Save processed feature matrix (encoded) sample
proc=df.copy()
proc.to_csv(f"{PROC}/streaming_subscribers_processed.csv",index=False)

# =====================================================================
# R1 REGULATORY & ANTITRUST  (scenario + Monte Carlo, losses in $B)
# Outcome probs reflect High likelihood of deep scrutiny.
# =====================================================================
NSIM=50000
p_clean,p_remedy,p_block=0.30,0.45,0.25
u=rng.random(NSIM)
loss_r1=np.empty(NSIM)
clean=u<p_clean; remedy=(u>=p_clean)&(u<p_clean+p_remedy); block=u>=p_clean+p_remedy
loss_r1[clean]=rng.triangular(0.0,0.2,0.6,clean.sum())                 # legal/process only
loss_r1[remedy]=rng.triangular(3,8,18,remedy.sum())                    # divestitures + delay
loss_r1[block]=rng.triangular(15,30,55,block.sum())                    # break fee + lost synergies
results["R1_expected_loss_B"]=round(float(loss_r1.mean()),2)
results["R1_sd_B"]=round(float(loss_r1.std()),2)
results["R1_p95_B"]=round(float(np.percentile(loss_r1,95)),2)
results["R1_p50_B"]=round(float(np.percentile(loss_r1,50)),2)
print(f"[R1] E[loss]=${loss_r1.mean():.2f}B  SD=${loss_r1.std():.2f}B  P95=${np.percentile(loss_r1,95):.2f}B")
fig,ax=plt.subplots(figsize=(6.4,4.4))
ax.hist(loss_r1,bins=60,color=NAVY,alpha=.85)
ax.axvline(loss_r1.mean(),color=GOLD,lw=2,label=f"Expected ${loss_r1.mean():.1f}B")
ax.axvline(np.percentile(loss_r1,95),color=RED,lw=2,ls="--",label=f"P95 ${np.percentile(loss_r1,95):.1f}B")
ax.set_title("R1 regulatory and antitrust: loss distribution"); ax.set_xlabel("Loss (USD billions)"); ax.set_ylabel("Frequency")
ax.legend(frameon=False,fontsize=9); style(ax)
plt.tight_layout(); plt.savefig(f"{FIG}/r1_regulatory_loss_mc.png",bbox_inches="tight"); plt.close()
pd.DataFrame({"loss_usd_billions":loss_r1[:5000]}).to_csv(f"{PROC}/r1_regulatory_simulation.csv",index=False)

# =====================================================================
# R2 REVENUE-AT-RISK FROM CHURN  (Monte Carlo, $)
# Assumptions: combined base 300M subs, ARPU $12/mo, baseline monthly churn
# 2.5%. Actual monthly churn drawn truncated-normal. First-year revenue loss
# approximates each incremental lost subscriber forgoing ~6 months of revenue.
# =====================================================================
subs=300e6; arpu=12.0; baseline=0.025
mc=np.clip(rng.normal(0.032,0.006,NSIM),baseline,0.08)
inc_monthly=mc-baseline
annual_subs_lost=subs*inc_monthly*12
rev_loss_r2=annual_subs_lost*arpu*6/1e9      # $B
results["R2_expected_revloss_B"]=round(float(rev_loss_r2.mean()),2)
results["R2_p95_revloss_B"]=round(float(np.percentile(rev_loss_r2,95)),2)
results["R2_expected_subs_lost_M"]=round(float(annual_subs_lost.mean()/1e6),1)
print(f"[R2] E[rev loss]=${rev_loss_r2.mean():.2f}B  P95=${np.percentile(rev_loss_r2,95):.2f}B  subs lost~{annual_subs_lost.mean()/1e6:.1f}M")
fig,ax=plt.subplots(figsize=(6.4,4.4))
ax.hist(rev_loss_r2,bins=60,color=TEAL,alpha=.85)
ax.axvline(rev_loss_r2.mean(),color=GOLD,lw=2,label=f"Expected ${rev_loss_r2.mean():.1f}B")
ax.axvline(np.percentile(rev_loss_r2,95),color=RED,lw=2,ls="--",label=f"P95 ${np.percentile(rev_loss_r2,95):.1f}B")
ax.set_title("R2 churn: first-year revenue at risk"); ax.set_xlabel("Revenue loss (USD billions)"); ax.set_ylabel("Frequency")
ax.legend(frameon=False,fontsize=9); style(ax)
plt.tight_layout(); plt.savefig(f"{FIG}/r2_revenue_loss_mc.png",bbox_inches="tight"); plt.close()

# =====================================================================
# R3 CYBERSECURITY  (FAIR-style Monte Carlo, annual loss $)
# Loss Event Frequency ~ Poisson(0.8). Per-event loss = records breached
# (lognormal) x cost per record (triangular) + response cost (lognormal).
# =====================================================================
lam=0.8
ale=np.zeros(NSIM)
n_events=rng.poisson(lam,NSIM)
total_events=int(n_events.sum())
records=rng.lognormal(mean=np.log(1_000_000),sigma=0.85,size=total_events)
cost_per_rec=rng.triangular(120,165,210,total_events)
response=rng.lognormal(mean=np.log(15e6),sigma=0.6,size=total_events)
event_loss=records*cost_per_rec+response
idx=0
for i,k in enumerate(n_events):
    if k>0:
        ale[i]=event_loss[idx:idx+k].sum(); idx+=k
ale_B=ale/1e9
results["R3_expected_ale_B"]=round(float(ale_B.mean()),3)
results["R3_p95_ale_B"]=round(float(np.percentile(ale_B,95)),3)
results["R3_p99_ale_B"]=round(float(np.percentile(ale_B,99)),3)
print(f"[R3] E[ALE]=${ale_B.mean():.3f}B  P95=${np.percentile(ale_B,95):.3f}B  P99=${np.percentile(ale_B,99):.3f}B")
fig,ax=plt.subplots(figsize=(6.4,4.4))
ax.hist(ale_B[ale_B>0],bins=60,color="#6A4C93",alpha=.85)
ax.axvline(ale_B.mean(),color=GOLD,lw=2,label=f"Expected ${ale_B.mean():.2f}B")
ax.axvline(np.percentile(ale_B,95),color=RED,lw=2,ls="--",label=f"P95 ${np.percentile(ale_B,95):.2f}B")
ax.set_title("R3 cybersecurity: annualized loss exposure"); ax.set_xlabel("Annual loss (USD billions)"); ax.set_ylabel("Frequency")
ax.legend(frameon=False,fontsize=9); style(ax)
plt.tight_layout(); plt.savefig(f"{FIG}/r3_cyber_ale_mc.png",bbox_inches="tight"); plt.close()
pd.DataFrame({"annual_loss_usd_billions":ale_B[:5000]}).to_csv(f"{PROC}/r3_cyber_simulation.csv",index=False)

# =====================================================================
# RISK EXPOSURE SUMMARY  (expected loss vs P95 by risk)
# =====================================================================
summary=pd.DataFrame({
    "risk":["R1 Regulatory","R2 Churn","R3 Cyber"],
    "expected_loss_B":[results["R1_expected_loss_B"],results["R2_expected_revloss_B"],results["R3_expected_ale_B"]],
    "p95_loss_B":[results["R1_p95_B"],results["R2_p95_revloss_B"],results["R3_p95_ale_B"]]})
summary.to_csv(f"{PROC}/risk_exposure_summary.csv",index=False)
fig,ax=plt.subplots(figsize=(6.8,4.4))
x=np.arange(3); w=0.38
ax.bar(x-w/2,summary.expected_loss_B,w,label="Expected loss",color=NAVY)
ax.bar(x+w/2,summary.p95_loss_B,w,label="P95 (tail) loss",color=GOLD)
ax.set_xticks(x); ax.set_xticklabels(summary.risk)
ax.set_ylabel("USD billions"); ax.set_title("Quantified risk exposure by risk")
for i,(e,pp) in enumerate(zip(summary.expected_loss_B,summary.p95_loss_B)):
    ax.text(i-w/2,e,f"{e:.1f}",ha="center",va="bottom",fontsize=9)
    ax.text(i+w/2,pp,f"{pp:.1f}",ha="center",va="bottom",fontsize=9)
ax.legend(frameon=False,fontsize=9); style(ax)
plt.tight_layout(); plt.savefig(f"{FIG}/risk_exposure_summary.png",bbox_inches="tight"); plt.close()

# risk register (for the qualitative notebook)
pd.DataFrame({
    "risk_id":["R1","R2","R3"],
    "risk":["Regulatory and antitrust","Subscriber churn and competitive response","Cybersecurity and data integration"],
    "category":["External, regulatory","Business, financial","Technological, infosec"],
    "likelihood":[7,7,5],"impact":[9,8,8],"score":[63,56,40],"band":["High","High","Medium"]
}).to_csv(f"{RAW}/risk_register.csv",index=False)

json.dump(results,open(f"{PROC}/key_results.json","w"),indent=2)
print("\n[done] figures + data written. Key results:")
print(json.dumps(results,indent=2)[:1500])
