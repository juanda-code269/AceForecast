import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,brier_score_loss,log_loss,roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="RallyOdds",page_icon="🎾",layout="wide")
st.title("RallyOdds — time-aware tennis probability modeling")
st.caption("Historical model estimates, never guaranteed match outcomes.")


@st.cache_data
def demo(n=4500):
    rng=np.random.default_rng(81); players=[f"Demo Player {i:02}" for i in range(1,61)]
    innate={p:rng.normal() for p in players}; rows=[]
    for i,date in enumerate(pd.date_range("2012-01-01",periods=n,freq="D")):
        a,b=rng.choice(players,2,replace=False); surface=rng.choice(["Hard","Clay","Grass"],p=[.55,.32,.13])
        rank_diff=np.clip((innate[a]-innate[b])*28+rng.normal(0,18),-150,150)
        recent=(innate[a]-innate[b])*.12+rng.normal(0,.16); serve=(innate[a]-innate[b])*.025+rng.normal(0,.04)
        h2h=rng.normal((innate[a]-innate[b])*.08,.22); logit=.018*rank_diff+2.3*recent+6*serve+1.1*h2h+(.18 if surface=="Grass" else 0)
        p=1/(1+np.exp(-logit)); rows.append((date,a,b,surface,rank_diff,recent,serve,h2h,rng.binomial(1,p)))
    return pd.DataFrame(rows,columns=["date","player_a","player_b","surface","ranking_advantage","recent_form_advantage","serve_advantage","head_to_head_advantage","player_a_win"])


def load(upload):
    if upload is None:return demo(),"Synthetic tour demonstration (fictional players)"
    d=pd.read_csv(upload); mapping={}
    for canonical,choices in {"date":{"date","match_date"},"player_a":{"player_a","winner_name"},"player_b":{"player_b","loser_name"},"player_a_win":{"player_a_win","target","outcome"}}.items():
        found=next((c for c in d if c.lower() in choices),None)
        if found:mapping[found]=canonical
    d=d.rename(columns=mapping)
    if not {"date","player_a","player_b","player_a_win"}.issubset(d):raise ValueError("CSV needs date, two player columns, a binary outcome, and numeric pre-match features.")
    d.date=pd.to_datetime(d.date,errors="coerce");return d.dropna(subset=["date","player_a_win"]),f"Uploaded data: {upload.name}"


upload=st.sidebar.file_uploader("Optional match CSV",type="csv")
try:
    df,source=load(upload); features=[c for c in df.select_dtypes(include=np.number) if c!="player_a_win"]
    if len(df)<300 or len(features)<2:raise ValueError("Need 300 matches and at least two numeric pre-match features.")
    df=df.sort_values("date").dropna(subset=features);cut=int(len(df)*.75);train,test=df.iloc[:cut],df.iloc[cut:]
except Exception as exc:st.error(str(exc));st.stop()
models={"Logistic Regression":make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)),"Random Forest":RandomForestClassifier(n_estimators=300,min_samples_leaf=8,class_weight="balanced",random_state=42)}
results=[];fitted={}
for name,m in models.items():
    m.fit(train[features],train.player_a_win);p=m.predict_proba(test[features])[:,1];fitted[name]=(m,p)
    results.append({"Model":name,"Accuracy":accuracy_score(test.player_a_win,p>=.5),"ROC-AUC":roc_auc_score(test.player_a_win,p),"Brier ↓":brier_score_loss(test.player_a_win,p),"Log loss ↓":log_loss(test.player_a_win,p)})
choice=st.sidebar.selectbox("Active model",list(models));model,prob=fitted[choice]
tabs=st.tabs(["Evaluation","Player analysis","Matchup simulator","Explainability","Uncertainty"])
with tabs[0]:
    st.info(f"{source}. Chronological split: earliest 75% trains; latest 25% tests.")
    st.dataframe(pd.DataFrame(results).style.format({"Accuracy":"{:.3f}","ROC-AUC":"{:.3f}","Brier ↓":"{:.3f}","Log loss ↓":"{:.3f}"}),width="stretch")
    obs,pred=calibration_curve(test.player_a_win,prob,n_bins=10,strategy="quantile");cal=pd.DataFrame({"Predicted":pred,"Observed":obs})
    fig=px.line(cal,x="Predicted",y="Observed",markers=True,title="Calibration on later matches");fig.add_scatter(x=[0,1],y=[0,1],line_dash="dash",name="Perfect")
    st.plotly_chart(fig,width="stretch")
with tabs[1]:
    names=sorted(set(df.player_a.astype(str))|set(df.player_b.astype(str)));player=st.selectbox("Player",names)
    involved=df[(df.player_a.astype(str)==player)|(df.player_b.astype(str)==player)].copy()
    involved["win_for_selected"]=np.where(involved.player_a.astype(str)==player,involved.player_a_win,1-involved.player_a_win)
    if "surface" in involved:
        summary=involved.groupby("surface").agg(matches=("win_for_selected","size"),win_rate=("win_for_selected","mean")).reset_index()
        st.plotly_chart(px.bar(summary,x="surface",y="win_rate",text="matches",title="Historical record by surface"),width="stretch")
    recent=involved.sort_values("date").tail(25);st.metric("Recent 25-match win rate",f"{recent.win_for_selected.mean():.1%}")
with tabs[2]:
    c1,c2=st.columns(2);a=c1.selectbox("Player A",names,index=0);b=c2.selectbox("Player B",names,index=1)
    if a==b:st.warning("Choose different players.")
    values={}
    for i,f in enumerate(features):
        s=train[f];values[f]=[c1,c2][i%2].slider(f,float(s.quantile(.01)),float(s.quantile(.99)),float(s.median()))
    p=float(model.predict_proba(pd.DataFrame([values]))[0,1]);st.metric(f"Estimated probability: {a} defeats {b}",f"{p:.1%}")
    st.caption("Player names label the scenario; the probability is driven by the selected feature values.")
with tabs[3]:
    core=model.named_steps.get("logisticregression") if hasattr(model,"named_steps") else model
    importance=np.abs(core.coef_[0]) if hasattr(core,"coef_") else core.feature_importances_
    imp=pd.DataFrame({"Feature":features,"Importance":importance}).sort_values("Importance")
    st.plotly_chart(px.bar(imp,x="Importance",y="Feature",orientation="h",title="Global feature influence"),width="stretch")
with tabs[4]:
    st.markdown("""Predictions are less reliable for players with few observations, new playing conditions, injuries, changing form, and feature combinations far from the training sample. Rankings and recent-form features must be computed using information available **before** each match; otherwise leakage can make performance look unrealistically strong. The default data is synthetic and the player names are fictional. Public professional and personal match data should remain separate.""")

