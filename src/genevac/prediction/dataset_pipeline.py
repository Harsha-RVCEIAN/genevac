"""Dataset ingestion and preparation for real-world and synthetic trajectories."""
from __future__ import annotations
import json, math, random, pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from genevac.prediction.data_contract import TrajectoryCategory, TrajectoryObservation

@dataclass(frozen=True)
class DatasetWindow:
    dataset: str
    scene: str
    agent_id: str
    split: str
    reference_timestamp: float
    history: tuple
    future: tuple

def identify_dataset(path: Path) -> str:
    parts={p.lower() for p in path.parts}; name=path.name.lower()
    if "stanford" in parts or path.suffix.lower()==".pkl": return "SDD"
    if "hotel" in parts or "biwi_hotel" in name: return "HOTEL"
    if "univ" in parts or "students" in name or "uni_examples" in name: return "UNIV"
    if "zara1" in parts or "zara01" in name: return "ZARA1"
    if "zara2" in parts or "zara02" in name: return "ZARA2"
    if "eth" in parts or "biwi_eth" in name: return "ETH"
    return "UNKNOWN"

def inventory_datasets(root: str | Path) -> list[dict[str, Any]]:
    root=Path(root); rows=[]
    if not root.exists(): return rows
    for p in sorted(x for x in root.rglob("*") if x.is_file() and x.suffix.lower() in {".txt",".csv",".pkl",".pickle"}):
        split=next((s for s in ("train","val","test") if s in {q.lower() for q in p.parts}),"unspecified")
        row={"path":p.as_posix(),"dataset":identify_dataset(p),"split":split,"format":p.suffix.lower()[1:],"size_bytes":p.stat().st_size}
        if p.suffix.lower() in {".txt",".csv"}:
            with p.open(encoding="utf-8",errors="replace") as f: row["line_count"]=sum(bool(x.strip()) for x in f)
        rows.append(row)
    return rows

def load_eth_ucy(filepath: str | Path, *, seconds_per_source_frame: float=0.04, sampling_stride: int=1):
    """Parse frame, track, x, y. At 25 FPS, one frame=0.04s; 10-frame gaps=0.4s."""
    if not math.isfinite(seconds_per_source_frame) or seconds_per_source_frame<=0: raise ValueError("invalid frame duration")
    if type(sampling_stride) is not int or sampling_stride<1: raise ValueError("sampling_stride must be positive")
    p=Path(filepath)
    if not p.is_file(): raise FileNotFoundError(p)
    raw={}
    with p.open(encoding="utf-8") as f:
        for no,line in enumerate(f,1):
            fields=line.strip().replace(","," ").split()
            if not fields: continue
            if len(fields)<4: raise ValueError(f"{p}:{no}: expected frame track x y")
            try: frame=int(float(fields[0])); aid=str(int(float(fields[1]))); x=float(fields[2]); y=float(fields[3])
            except ValueError as e: raise ValueError(f"{p}:{no}: invalid numeric field") from e
            if not math.isfinite(x) or not math.isfinite(y): raise ValueError(f"{p}:{no}: non-finite coordinate")
            raw.setdefault(aid,[]).append((frame,x,y))
    out={}
    for aid,points in raw.items():
        points.sort()
        if len({v[0] for v in points})!=len(points): raise ValueError(f"duplicate frame for {aid}")
        points=points[::sampling_stride]
        if len(points)<2: continue
        t0=points[0][0]
        out[aid]=[TrajectoryObservation(agent_id=aid,step_index=i,timestamp=round((fr-t0)*seconds_per_source_frame,6),x=x,y=y,category=TrajectoryCategory.OBSERVED) for i,(fr,x,y) in enumerate(points)]
    return out

def load_sdd_pickle(filepath: str | Path, *, trusted_pickle: bool=False, seconds_per_frame: float=1.0):
    """Opt-in loader for trusted pickle files. Pickle can execute arbitrary code."""
    if not trusted_pickle: raise ValueError("pickle loading disabled; pass trusted_pickle=True only for verified files")
    p=Path(filepath)
    if not p.is_file(): raise FileNotFoundError(p)
    with p.open("rb") as f: obj=pickle.load(f)
    tracks={}
    if hasattr(obj,"columns") and hasattr(obj,"iterrows"):
        cols={str(c).lower():c for c in obj.columns}
        aliases={"id":("track_id","ped_id","agent_id","trackid"),"frame":("frame","frame_id","timestamp","time"),"x":("x","pos_x"),"y":("y","pos_y")}
        keys={k:next((cols[a] for a in v if a in cols),None) for k,v in aliases.items()}
        if all(keys.values()):
            for _,r in obj.iterrows(): tracks.setdefault(str(r[keys["id"]]),[]).append((float(r[keys["frame"]]),float(r[keys["x"]]),float(r[keys["y"]])))
    if not tracks and isinstance(obj,dict):
        for aid,seq in obj.items():
            try:
                arr=list(seq)
                if arr and len(arr[0])>=3: tracks[str(aid)]=[(float(v[0]),float(v[-2]),float(v[-1])) for v in arr]
            except (TypeError,ValueError,IndexError): pass
    if not tracks: raise ValueError("unsupported SDD pickle schema; convert to track_id/frame/x/y table")
    out={}
    for aid,seq in tracks.items():
        seq.sort()
        if len(seq)<2: continue
        t0=seq[0][0]
        out[aid]=[TrajectoryObservation(agent_id=aid,step_index=i,timestamp=(fr-t0)*seconds_per_frame,x=x,y=y,category=TrajectoryCategory.OBSERVED) for i,(fr,x,y) in enumerate(seq)]
    return out

def validate_trajectories(tracks: Mapping[str,Sequence[TrajectoryObservation]]):
    errors=[]; count=0
    for aid,seq in tracks.items():
        prev=None
        for o in seq:
            count+=1
            if o.agent_id!=aid: errors.append(f"{aid}: id mismatch")
            if not all(math.isfinite(float(v)) for v in (o.timestamp,o.x,o.y)): errors.append(f"{aid}: non-finite value")
            if prev and (o.timestamp<=prev.timestamp or o.step_index<=prev.step_index): errors.append(f"{aid}: non-monotonic sequence")
            prev=o
    return {"valid":not errors,"agents":len(tracks),"observations":count,"errors":errors}

def split_by_agent(tracks: Mapping[str,Sequence], *, seed=42, train=.7, validation=.15, test=.15):
    ratios=(train,validation,test)
    if any(v<0 for v in ratios) or not math.isclose(sum(ratios),1,abs_tol=1e-8): raise ValueError("split ratios must sum to 1")
    ids=sorted(k for k,v in tracks.items() if len(v)>=2); random.Random(seed).shuffle(ids)
    a=int(len(ids)*train); b=a+int(len(ids)*validation)
    return {"train":ids[:a],"validation":ids[a:b],"test":ids[b:]}

def make_windows(tracks, splits, *, history_length=8, prediction_horizon=12, stride=1, dataset="unknown", scene="unknown"):
    if min(history_length,prediction_horizon,stride)<1: raise ValueError("window sizes must be positive")
    out=[]
    for split,ids in splits.items():
        for aid in ids:
            seq=tracks[aid]
            for start in range(0,len(seq)-history_length-prediction_horizon+1,stride):
                h=seq[start:start+history_length]; f=seq[start+history_length:start+history_length+prediction_horizon]
                out.append(DatasetWindow(dataset,scene,aid,split,float(h[-1].timestamp),tuple((o.timestamp,o.x,o.y) for o in h),tuple((o.timestamp,o.x,o.y) for o in f)))
    return out

def export_dataset(tracks, output_dir, *, dataset, scene, source, seed=42, history_length=8, prediction_horizon=12, fixed_split=None):
    check=validate_trajectories(tracks)
    if not check["valid"]: raise ValueError("invalid trajectories: "+"; ".join(check["errors"][:5]))
    if fixed_split is not None:
        if fixed_split not in {"train","validation","test"}: raise ValueError("fixed_split must be train, validation or test")
        splits={fixed_split:sorted(k for k,v in tracks.items() if len(v)>=2)}
    else:
        splits=split_by_agent(tracks,seed=seed)
    windows=make_windows(tracks,splits,history_length=history_length,prediction_horizon=prediction_horizon,dataset=dataset,scene=scene)
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    for name in ("train","validation","test"):
        with (out/f"{dataset.lower()}_{scene.lower()}_{name}.jsonl").open("w",encoding="utf-8") as f:
            for w in windows:
                if w.split==name: f.write(json.dumps(asdict(w))+"\n")
    manifest={"schema_version":"1.0","dataset":dataset,"scene":scene,"source":str(source),"coordinate_units":"metres","temporal_units":"seconds","split_strategy":"official_source_partition" if fixed_split else "agent_disjoint","source_split":fixed_split,"seed":seed,"history_length":history_length,"prediction_horizon":prediction_horizon,"validation":check,"split_agents":splits,"window_counts":{k:sum(w.split==k for w in windows) for k in ("train","validation","test")},"note":"Keep real-world benchmark scores separate from synthetic evacuation scores."}
    (out/f"{dataset.lower()}_{scene.lower()}_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    return manifest

class SimulationDatasetRecorder:
    """Capture trajectories from the existing authoritative SimulationEngine."""
    def __init__(self, observation_interval=0.1):
        from genevac.prediction.dataset_adapter import SimulationEvaluationAdapter
        self.adapter=SimulationEvaluationAdapter(observation_interval=observation_interval); self.steps=0
    def capture(self, engine):
        self.adapter.step(engine); self.steps+=1
    def export(self, output_dir, *, scenario_id, seed=None, run_id="run"):
        tracks=self.adapter.get_trajectories()
        manifest=export_dataset(tracks,output_dir,dataset="GENEVAC_SYNTHETIC",scene=scenario_id,source="GenEvac SimulationEngine")
        manifest.update({"scenario_id":scenario_id,"seed":seed,"run_id":run_id,"captured_steps":self.steps})
        Path(output_dir).mkdir(parents=True,exist_ok=True)
        Path(output_dir,f"{run_id}_metadata.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
        return manifest
