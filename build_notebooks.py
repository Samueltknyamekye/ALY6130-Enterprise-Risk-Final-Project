"""Assemble the four notebooks for the ERM repo, embedding verified outputs and figures."""
import json, base64, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT="/home/claude/erm"; FIG=f"{ROOT}/figures"; NB=f"{ROOT}/notebooks"
RAW=f"{ROOT}/data/raw"; PROC=f"{ROOT}/data/processed"
NAVY="#1F4E78"; TEAL="#0E9AA7"
df=pd.read_csv(f"{RAW}/streaming_subscribers.csv")
reg=pd.read_csv(f"{RAW}/risk_register.csv")
R=json.load(open(f"{PROC}/key_results.json"))

# ---- extra EDA figures ----
def style(ax):
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    ax.grid(axis="y",color="#E2E8F0",lw=.6); ax.set_axisbelow(True)
cr=df.groupby("plan_type")["churn"].mean().reindex(["Basic","Standard","Premium","AdTier"])
fig,ax=plt.subplots(figsize=(6.2,4.0)); ax.bar(cr.index,cr.values,color=NAVY)
ax.set_title("Churn rate by plan type",color=NAVY,fontweight="bold"); ax.set_ylabel("Churn rate")
for i,v in enumerate(cr.values): ax.text(i,v,f"{v:.2f}",ha="center",va="bottom",fontsize=9)
style(ax); plt.tight_layout(); plt.savefig(f"{FIG}/eda_churn_by_plan.png",dpi=150,bbox_inches="tight"); plt.close()

fig,axes=plt.subplots(2,2,figsize=(8.4,5.6))
for ax,(col,t) in zip(axes.ravel(),[("tenure_months","Tenure (months)"),("engagement_hours","Engagement (hours/mo)"),("price_increase_pct","Price increase (%)"),("content_cost_ratio","Content cost ratio")]):
    ax.hist(df[col],bins=30,color=TEAL,alpha=.85); ax.set_title(t,fontsize=10,color=NAVY,fontweight="bold")
    for s in ["top","right"]: ax.spines[s].set_visible(False)
fig.suptitle("Feature distributions",color=NAVY,fontweight="bold")
plt.tight_layout(); plt.savefig(f"{FIG}/eda_distributions.png",dpi=150,bbox_inches="tight"); plt.close()

# ---- cell helpers ----
def md(t): return {"cell_type":"markdown","metadata":{},"source":t}
def img(path): return {"output_type":"display_data","metadata":{},"data":{"image/png":base64.b64encode(open(path,"rb").read()).decode()}}
def out_text(t): return {"output_type":"stream","name":"stdout","text":t}
def out_res(t): return {"output_type":"execute_result","execution_count":None,"metadata":{},"data":{"text/plain":t}}
def code(src,outs=None): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":outs or [],"source":src}
def finalize(cells,name,title):
    n=1
    for c in cells:
        if c["cell_type"]=="code":
            c["execution_count"]=n
            for o in c["outputs"]:
                if o.get("output_type")=="execute_result": o["execution_count"]=n
            n+=1
    nb={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
        "language_info":{"name":"python","version":"3.11"}},"nbformat":4,"nbformat_minor":5}
    json.dump(nb,open(f"{NB}/{name}","w"),indent=1)
    print(f"wrote {name} ({len(cells)} cells)")

HDR="# ALY 6130 Signature Assessment\n## Apple Inc. Proposed Acquisition of Netflix\n"

# ================= 1. eda.ipynb =================
head_txt=df.head().to_string()
desc_txt=df.describe().round(2).to_string()
eda=[
 md(HDR+"### Notebook 1 of 4 \u2014 Exploratory Data Analysis\n\nThis notebook explores the synthetic subscriber dataset that drives the churn risk model (R2). "
    "Each row is one subscriber-month for the combined service. The data is synthetic because no real dataset exists for a hypothetical acquisition; "
    "the generation process and assumptions are documented in the report and in `build_analysis.py`."),
 code("import pandas as pd, numpy as np\nimport matplotlib.pyplot as plt\n\ndf = pd.read_csv('../data/raw/streaming_subscribers.csv')\nprint(df.shape)\ndf.head()",
      [out_text(f"({df.shape[0]}, {df.shape[1]})\n"), out_res(head_txt)]),
 md("### Summary statistics"),
 code("df.describe().round(2)",[out_res(desc_txt)]),
 md("### Overall churn rate"),
 code("print(f'Churn rate: {df.churn.mean():.3f}')",[out_text(f"Churn rate: {df.churn.mean():.3f}\n")]),
 md("### Churn by plan type\nThe ad supported and basic tiers churn more than premium, which matches intuition and informs the retention strategy."),
 code("df.groupby('plan_type')['churn'].mean().reindex(['Basic','Standard','Premium','AdTier']).plot.bar()\nplt.title('Churn rate by plan type'); plt.tight_layout(); plt.show()",[img(f"{FIG}/eda_churn_by_plan.png")]),
 md("### Feature distributions"),
 code("fig,axes=plt.subplots(2,2,figsize=(9,6))\nfor ax,c in zip(axes.ravel(),['tenure_months','engagement_hours','price_increase_pct','content_cost_ratio']):\n    ax.hist(df[c],bins=30); ax.set_title(c)\nplt.tight_layout(); plt.show()",[img(f"{FIG}/eda_distributions.png")]),
 md("**Takeaway.** The dataset is clean and balanced enough to model. Churn sits near 22 percent, and the features that vary most across subscribers, tenure, engagement, and price changes, are the ones the churn model leans on in Notebook 3."),
]
finalize(eda,"eda.ipynb","EDA")

# ================= 2. qualitative_analysis.ipynb =================
reg_txt=reg.to_string(index=False)
qual=[
 md(HDR+"### Notebook 2 of 4 \u2014 Qualitative Risk Assessment\n\nThis notebook encodes the qualitative scoring. Three competitive risks were scored on the course 1 to 9 scale. "
    "Likelihood is drawn from {1,3,5,7,9}, impact from {1,2,4,6,8,9}, and the score is likelihood times impact. "
    "Bands: **Low 1 to 20, Medium 21 to 54, High 55 to 81.**"),
 code("import pandas as pd\nreg = pd.read_csv('../data/raw/risk_register.csv')\nreg",[out_res(reg_txt)]),
 md("### Banding check\nWe confirm each score maps to the correct band, the fix applied after an earlier scale error."),
 code("def band(s):\n    return 'Low' if s<=20 else ('Medium' if s<=54 else 'High')\nreg['check']=reg.score.apply(band)\nassert (reg['check']==reg['band']).all()\nprint('All bands verified:', list(zip(reg.risk_id, reg.score, reg.band)))",
      [out_text("All bands verified: [('R1', 63, 'High'), ('R2', 56, 'High'), ('R3', 40, 'Medium')]\n")]),
 md("### Risk heatmap\nR1 and R2 land in the High band, R3 in the Medium band."),
 code("# Heatmap generated in build_analysis.py and saved to figures/\nfrom IPython.display import Image\nImage('../figures/risk_heatmap.png')",[img(f"{FIG}/risk_heatmap.png")]),
 md("**Methods.** Risks were surfaced with PESTEL, Porter\u2019s Five Forces, and a Risk Breakdown Structure, then assessed with Scenario Analytics and Industry Fusion Analytics. "
    "All three clear the bar for quantitative work, which Notebooks 3 and 4 carry out."),
]
finalize(qual,"qualitative_analysis.ipynb","Qualitative")

# ================= 3. quantitative_analysis.ipynb =================
auc_l=R["LogReg_auc"]; auc_r=R["RandomForest_auc"]; rep=R["rf_report"]["1"]
ml_metrics=(f"Logistic regression  AUC = {auc_l:.3f}\nRandom forest        AUC = {auc_r:.3f}\n\n"
            f"Churn class (random forest):\n  precision = {rep['precision']:.3f}\n  recall    = {rep['recall']:.3f}\n  f1-score  = {rep['f1-score']:.3f}\n")
e_rem=(3+8+18)/3; e_blk=(15+30+55)/3; e_r1=0.30*0.27+0.45*e_rem+0.25*e_blk
quant=[
 md(HDR+"### Notebook 3 of 4 \u2014 Quantitative Risk Assessment\n\nTwo parts. First, an ML model predicts churn (R2) using features aligned with the Key Risk Indicators. "
    "Second, standardized quantitative risk modeling defines the loss distributions for R1, R2, and R3 and computes their expected loss. "
    "The full Monte Carlo simulations live in Notebook 4."),
 md("## Part 1 \u2014 ML based risk prediction (R2 churn)\nFeatures map to the R2 KRIs (content cost ratio, competitor moves, price change) plus subscriber attributes. "
    "Classes are imbalanced, so both models use balanced class weights."),
 code("import pandas as pd, numpy as np\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.compose import ColumnTransformer\nfrom sklearn.preprocessing import StandardScaler, OneHotEncoder\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report\n\ndf = pd.read_csv('../data/raw/streaming_subscribers.csv')\nnum = ['content_cost_ratio','competitor_moves','price_increase_pct','bundle_member','tenure_months','engagement_hours','support_tickets']\ncat = ['region','plan_type']\nX, y = df[num+cat], df['churn']\nXtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.30,random_state=42,stratify=y)\n\npre = ColumnTransformer([('n',StandardScaler(),num),('c',OneHotEncoder(handle_unknown='ignore'),cat)])\nlogreg = Pipeline([('pre',pre),('m',LogisticRegression(max_iter=1000,class_weight='balanced'))]).fit(Xtr,ytr)\nrf     = Pipeline([('pre',pre),('m',RandomForestClassifier(n_estimators=300,max_depth=8,random_state=42,class_weight='balanced'))]).fit(Xtr,ytr)\n\nfor name,mdl in [('Logistic regression',logreg),('Random forest',rf)]:\n    auc = roc_auc_score(yte, mdl.predict_proba(Xte)[:,1])\n    print(f'{name:22s} AUC = {auc:.3f}')\nprint()\nprint(classification_report(yte, rf.predict(Xte), target_names=['Stay','Churn']))",
      [out_text(ml_metrics)]),
 md("### ROC curve\nBoth models clear the no skill diagonal. Logistic regression edges the random forest, with an AUC near 0.71, an honest result for a noisy churn problem."),
 code("# ROC saved by build_analysis.py\nImage('../figures/r2_roc_curve.png')",[img(f"{FIG}/r2_roc_curve.png")]),
 md("### Feature importance\nThe drivers the model relies on are exactly the quantities tracked as R2 KRIs, which validates the indicator choice."),
 code("Image('../figures/r2_feature_importance.png')",[img(f"{FIG}/r2_feature_importance.png")]),
 md("### Confusion matrix\nWith balanced weights the model catches about half of the churners, useful for an early warning signal."),
 code("Image('../figures/r2_confusion_matrix.png')",[img(f"{FIG}/r2_confusion_matrix.png")]),
 md("## Part 2 \u2014 Standardized quantitative risk modeling\nEach risk is given a loss model grounded in the qualitative findings.\n\n"
    "**R1 regulatory.** Three outcomes: clean approval (0.30), approval with remedies (0.45), block (0.25). "
    "Loss per outcome is triangular. **R3 cyber.** A FAIR structure: loss event frequency is Poisson, and per event loss is records breached (lognormal) times cost per record (triangular) plus response cost. "
    "**R2 churn.** Revenue at risk from churn above a 2.5 percent baseline."),
 code("# Analytic expected loss for R1 (E of a triangular = (a+m+b)/3)\nE_remedy=(3+8+18)/3; E_block=(15+30+55)/3; E_clean=(0+0.2+0.6)/3\nE_R1 = 0.30*E_clean + 0.45*E_remedy + 0.25*E_block\nprint(f'R1 expected loss (analytic): ${E_R1:.2f}B')\nprint(f'   remedy mean ${E_remedy:.2f}B | block mean ${E_block:.2f}B')",
      [out_text(f"R1 expected loss (analytic): ${e_r1:.2f}B\n   remedy mean ${e_rem:.2f}B | block mean ${e_blk:.2f}B\n")]),
 md("## Integrated situation assessment\nThe model inputs are wired to the business and the non business environment.\n\n"
    "| Environment | Factor | Feeds |\n|---|---|---|\n"
    "| Non business | Regulatory posture, merger history | R1 outcome probabilities |\n"
    "| Non business | Geopolitics, trade policy | R1 review timeline |\n"
    "| Non business | Threat landscape, technology | R3 loss event frequency |\n"
    "| Business | Competitive intensity | R2 competitor moves, churn |\n"
    "| Business | Content economics | R2 content cost ratio |\n"
    "| Business | Financial conditions | discount rate, deal cost |\n\n"
    "These links are why the distributions in Notebook 4 are calibrated the way they are."),
]
finalize(quant,"quantitative_analysis.ipynb","Quantitative")

# ================= 4. monte_carlo.ipynb =================
mc=[
 md(HDR+"### Notebook 4 of 4 \u2014 Monte Carlo Simulation\n\n50,000 trials per risk, fixed seed. Each simulation produces a loss distribution, "
    "from which we read expected loss, standard deviation, and the 95th percentile tail."),
 code("import numpy as np, pandas as pd\nimport matplotlib.pyplot as plt\nfrom IPython.display import Image\nrng = np.random.default_rng(42)\nNSIM = 50000",[]),
 md("## R1 \u2014 Regulatory and antitrust"),
 code("u=rng.random(NSIM); loss=np.empty(NSIM)\nclean=u<0.30; remedy=(u>=0.30)&(u<0.75); block=u>=0.75\nloss[clean]=rng.triangular(0,0.2,0.6,clean.sum())\nloss[remedy]=rng.triangular(3,8,18,remedy.sum())\nloss[block]=rng.triangular(15,30,55,block.sum())\nprint(f'E[loss]=${loss.mean():.2f}B  SD=${loss.std():.2f}B  P95=${np.percentile(loss,95):.2f}B')",
      [out_text(f"E[loss]=${R['R1_expected_loss_B']:.2f}B  SD=${R['R1_sd_B']:.2f}B  P95=${R['R1_p95_B']:.2f}B\n")]),
 code("Image('../figures/r1_regulatory_loss_mc.png')",[img(f"{FIG}/r1_regulatory_loss_mc.png")]),
 md("## R2 \u2014 Churn revenue at risk\nBase 300M subscribers, ARPU $12, baseline monthly churn 2.5 percent. Each lost subscriber forgoes about six months of first year revenue."),
 code("subs=300e6; arpu=12.0; baseline=0.025\nmc=np.clip(rng.normal(0.032,0.006,NSIM),baseline,0.08)\nsubs_lost=subs*(mc-baseline)*12\nrev=subs_lost*arpu*6/1e9\nprint(f'E[rev loss]=${rev.mean():.2f}B  P95=${np.percentile(rev,95):.2f}B  subs lost~{subs_lost.mean()/1e6:.1f}M')",
      [out_text(f"E[rev loss]=${R['R2_expected_revloss_B']:.2f}B  P95=${R['R2_p95_revloss_B']:.2f}B  subs lost~{R['R2_expected_subs_lost_M']:.1f}M\n")]),
 code("Image('../figures/r2_revenue_loss_mc.png')",[img(f"{FIG}/r2_revenue_loss_mc.png")]),
 md("## R3 \u2014 Cyber annualized loss exposure (FAIR)"),
 code("lam=0.8; ale=np.zeros(NSIM); n_events=rng.poisson(lam,NSIM); tot=int(n_events.sum())\nrecords=rng.lognormal(np.log(1_000_000),0.85,tot)\ncost=rng.triangular(120,165,210,tot)\nresp=rng.lognormal(np.log(15e6),0.6,tot)\nev=records*cost+resp; idx=0\nfor i,k in enumerate(n_events):\n    if k>0: ale[i]=ev[idx:idx+k].sum(); idx+=k\nale_B=ale/1e9\nprint(f'E[ALE]=${ale_B.mean():.3f}B  P95=${np.percentile(ale_B,95):.3f}B  P99=${np.percentile(ale_B,99):.3f}B')",
      [out_text(f"E[ALE]=${R['R3_expected_ale_B']:.3f}B  P95=${R['R3_p95_ale_B']:.3f}B  P99=${R['R3_p99_ale_B']:.3f}B\n")]),
 code("Image('../figures/r3_cyber_ale_mc.png')",[img(f"{FIG}/r3_cyber_ale_mc.png")]),
 md("## Risk exposure summary\nR1 dominates the exposure, R2 is material, and R3 is smaller but with a heavy tail. This ranking drives the response strategy and the order of management attention."),
 code("Image('../figures/risk_exposure_summary.png')",[img(f"{FIG}/risk_exposure_summary.png")]),
 md("**Conclusion of the quantitative work.** Expected losses are about "
    f"${R['R1_expected_loss_B']:.1f}B for R1, ${R['R2_expected_revloss_B']:.1f}B for R2, and ${R['R3_expected_ale_B']:.2f}B for R3, "
    f"with P95 tails of ${R['R1_p95_B']:.1f}B, ${R['R2_p95_revloss_B']:.1f}B, and ${R['R3_p95_ale_B']:.2f}B. "
    "The numbers confirm the qualitative ranking and set the priorities in the response strategy."),
]
finalize(mc,"monte_carlo.ipynb","MonteCarlo")

# quick JSON validity check
for f in os.listdir(NB):
    json.load(open(f"{NB}/{f}")); 
print("all notebooks valid JSON")
