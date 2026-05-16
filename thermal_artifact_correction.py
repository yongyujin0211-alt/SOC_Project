import numpy as np
from scipy.ndimage import gaussian_filter
import warnings, os

def _radial_map(rows, cols):
    cy, cx = (rows-1)/2.0, (cols-1)/2.0
    y, x = np.mgrid[0:rows, 0:cols]
    return np.sqrt(((x-cx)/max(cx,1))**2 + ((y-cy)/max(cy,1))**2)

def _robust_bin_median(v, iqr_factor=1.5):
    if len(v)==0: return np.nan
    q25,q75 = np.percentile(v,[25,75])
    iqr = q75-q25
    f = v[(v>=q25-iqr_factor*iqr)&(v<=q75+iqr_factor*iqr)]
    return float(np.median(f)) if len(f)>0 else float(np.median(v))

class VignettingCorrector:
    def __init__(self,rows,cols,vignetting_n_bins=24,vignetting_poly_degree=4,vignetting_min_pixels=5):
        self.rows,self.cols=rows,cols
        self.r_map=_radial_map(rows,cols)
        self.n_bins=vignetting_n_bins; self.poly_degree=vignetting_poly_degree
        self.min_pixels=vignetting_min_pixels
        self._flat_map=None; self._blind_offset=None
    def calibrate_from_flatfield(self,T_flat):
        cy,cx=self.rows//2,self.cols//2
        T_ref=np.mean(T_flat[cy-2:cy+3,cx-2:cx+3])
        if abs(T_ref)<1e-6: raise ValueError("T_ref~0")
        self._flat_map=T_flat/T_ref; return self._flat_map
    def _estimate_blind(self,T_raw):
        r_flat=self.r_map.ravel(); T_flat=T_raw.ravel()
        r_edges=np.linspace(0.0,r_flat.max(),self.n_bins+1)
        rc,tm=[],[]
        for i in range(self.n_bins):
            mask=(r_flat>=r_edges[i])&(r_flat<r_edges[i+1])
            if mask.sum()>=self.min_pixels:
                med=_robust_bin_median(T_flat[mask])
                if not np.isnan(med): rc.append((r_edges[i]+r_edges[i+1])/2); tm.append(med)
        ra,ta=np.array(rc),np.array(tm)
        if len(ra)<self.poly_degree+1:
            warnings.warn("[Vignetting] 유효구간 부족"); self._blind_offset=np.zeros_like(T_raw); return self._blind_offset
        co=np.polyfit(ra,ta,self.poly_degree)
        self._blind_offset=np.polyval(co,self.r_map)-np.polyval(co,0.0); return self._blind_offset
    def correct(self,T_raw):
        if self._flat_map is not None: return T_raw/np.where(self._flat_map>1e-6,self._flat_map,1.0)
        self._estimate_blind(T_raw); return T_raw-self._blind_offset

class FPNCorrector:
    """[수정] bilateral_sigma_color: 0.5->2.5 도씨 (야외 방사율 노이즈 현실화)"""
    def __init__(self,rows,cols,col_smooth_sigma=2.0,row_smooth_sigma=2.0,
                 pixel_smooth_sigma=2.5,fpn_sigma_threshold=4.0,
                 bilateral_d=5,bilateral_sigma_color=2.5,bilateral_sigma_space=2.0):
        self.rows,self.cols=rows,cols
        self.cs=col_smooth_sigma; self.rs=row_smooth_sigma; self.ps=pixel_smooth_sigma
        self.thr=fpn_sigma_threshold; self.bd=bilateral_d
        self.bc=bilateral_sigma_color; self.bs=bilateral_sigma_space
        self._gain_map=None; self._off_map=None
    def calibrate_two_point(self,Tcf,Thf,Tc_ref,Th_ref):
        Tc=np.mean(Tcf,axis=0); Th=np.mean(Thf,axis=0)
        dT=Th_ref-Tc_ref
        if abs(dT)<0.1: raise ValueError("온도차<0.1")
        self._gain_map=(Th-Tc)/dT
        self._gain_map=np.where(np.abs(self._gain_map)>1e-4,self._gain_map,1.0)
        self._off_map=Tc-self._gain_map*Tc_ref
    def correct(self,T_raw):
        if self._gain_map is not None and self._off_map is not None:
            return (T_raw-self._off_map)/self._gain_map
        T=T_raw-gaussian_filter(np.median(T_raw,axis=0)-np.median(T_raw),sigma=self.cs)[np.newaxis,:]
        T=T-gaussian_filter(np.median(T,axis=1)-np.median(T),sigma=self.rs)[:,np.newaxis]
        try:
            import cv2
            Ts=cv2.bilateralFilter(T.astype(np.float32),d=self.bd,sigmaColor=float(self.bc),sigmaSpace=float(self.bs)).astype(np.float64)
        except ImportError:
            warnings.warn("[FPN] OpenCV없음->Gaussian폴백",stacklevel=3); Ts=gaussian_filter(T,sigma=self.ps)
        res=T-Ts; mask=np.abs(res)>self.thr*np.std(res)
        Tc2=T.copy(); Tc2[mask]=Ts[mask]; return Tc2

class NarcissusCorrector:
    """[수정] 완전구현 - 반경방향 다항식 피팅 기반 (기존 Truncated 상태 수정)"""
    def __init__(self,rows,cols,narcissus_n_bins=32,background_poly_degree=2,ring_sigma=1.0):
        self.rows,self.cols=rows,cols
        self.r_map=_radial_map(rows,cols)
        self.n_bins=narcissus_n_bins; self.bg_degree=background_poly_degree; self.ring_sigma=ring_sigma
        self._narcissus_map=None
    def estimate(self,T):
        rf=self.r_map.ravel(); Tf=T.ravel()
        re=np.linspace(0.0,rf.max(),self.n_bins+1); rc,tm=[],[]
        for i in range(self.n_bins):
            mask=(rf>=re[i])&(rf<re[i+1])
            if mask.sum()<4: continue
            med=_robust_bin_median(Tf[mask])
            if not np.isnan(med): rc.append((re[i]+re[i+1])/2); tm.append(med)
        ra,ta=np.array(rc),np.array(tm)
        if len(ra)<self.bg_degree+2:
            warnings.warn("[Narcissus] 유효구간 부족->스킵"); self._narcissus_map=np.zeros_like(T); return self._narcissus_map
        bg=np.polyfit(ra,ta,self.bg_degree)
        narc=ta-np.polyval(bg,ra)
        if self.ring_sigma>0: narc=gaussian_filter(narc.astype(np.float64),sigma=self.ring_sigma)
        self._narcissus_map=np.interp(self.r_map.ravel(),ra,narc).reshape(self.rows,self.cols)
        return self._narcissus_map
    def correct(self,T):
        self.estimate(T); return T-self._narcissus_map
    @property
    def narcissus_map(self): return self._narcissus_map

class ThermalPreprocessor:
    """통합 전처리기 - 완전구현. 순서: FPN -> Vignetting -> Narcissus"""
    def __init__(self,rows,cols,flat_field=None,dark_frames=None,
                 enable_fpn=True,enable_vignetting=True,enable_narcissus=True,
                 vignetting_n_bins=24,vignetting_poly_degree=4,vignetting_min_pixels=5,
                 col_smooth_sigma=2.0,row_smooth_sigma=2.0,pixel_smooth_sigma=2.5,
                 fpn_sigma_threshold=4.0,bilateral_d=5,
                 bilateral_sigma_color=2.5,bilateral_sigma_space=2.0,
                 narcissus_n_bins=32,narcissus_bg_poly=2,narcissus_ring_sigma=1.0):
        self.rows,self.cols=rows,cols
        self.en_fpn=enable_fpn; self.en_vig=enable_vignetting; self.en_nar=enable_narcissus
        self.fpn=FPNCorrector(rows,cols,col_smooth_sigma=col_smooth_sigma,
            row_smooth_sigma=row_smooth_sigma,pixel_smooth_sigma=pixel_smooth_sigma,
            fpn_sigma_threshold=fpn_sigma_threshold,bilateral_d=bilateral_d,
            bilateral_sigma_color=bilateral_sigma_color,bilateral_sigma_space=bilateral_sigma_space)
        self.vig=VignettingCorrector(rows,cols,vignetting_n_bins=vignetting_n_bins,
            vignetting_poly_degree=vignetting_poly_degree,vignetting_min_pixels=vignetting_min_pixels)
        self.nar=NarcissusCorrector(rows,cols,narcissus_n_bins=narcissus_n_bins,
            background_poly_degree=narcissus_bg_poly,ring_sigma=narcissus_ring_sigma)
        if flat_field is not None: self.vig.calibrate_from_flatfield(flat_field)
        self._log=[]
    def correct(self,T_raw):
        if T_raw.shape!=(self.rows,self.cols): raise ValueError(f"크기불일치 {T_raw.shape}")
        T=T_raw.copy(); self._log=[]
        if self.en_fpn:
            Tb=T.copy(); T=self.fpn.correct(T)
            self._log.append({"step":"FPN (sig_c=2.5)","dm":float(np.mean(np.abs(T-Tb))),"dx":float(np.max(np.abs(T-Tb)))})
        if self.en_vig:
            Tb=T.copy(); T=self.vig.correct(T)
            self._log.append({"step":"Vignetting","dm":float(np.mean(np.abs(T-Tb))),"dx":float(np.max(np.abs(T-Tb)))})
        if self.en_nar:
            Tb=T.copy(); T=self.nar.correct(T)
            self._log.append({"step":"Narcissus (radial poly-fit)","dm":float(np.mean(np.abs(T-Tb))),"dx":float(np.max(np.abs(T-Tb)))})
        return T
    def correction_summary(self):
        lines=["[전처리 보정 요약]"]
        for e in self._log: lines.append(f"  {e['step']:35s}| dm={e['dm']:.4f}C  dx={e['dx']:.4f}C")
        if not self._log: lines.append("  (미실행)")
        return "\n".join(lines)
    def visualize_correction(self,T_raw,T_corrected,save_path="./correction.png"):
        try: import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        except ImportError: warnings.warn("matplotlib없음"); return ""
        fig,axes=plt.subplots(2,2,figsize=(13,10)); diff=T_raw-T_corrected
        im=axes[0,0].imshow(T_raw,cmap="inferno",aspect="equal")
        axes[0,0].set_title("(a) 원본 [C]"); plt.colorbar(im,ax=axes[0,0],shrink=0.85)
        im2=axes[0,1].imshow(T_corrected,cmap="inferno",aspect="equal",vmin=T_raw.min(),vmax=T_raw.max())
        axes[0,1].set_title("(b) 보정후 [C]"); plt.colorbar(im2,ax=axes[0,1],shrink=0.85)
        abm=max(abs(diff.min()),abs(diff.max()),0.01)
        im3=axes[1,0].imshow(diff,cmap="RdBu_r",aspect="equal",vmin=-abm,vmax=abm)
        axes[1,0].set_title("(c) 왜곡차분 [C]"); plt.colorbar(im3,ax=axes[1,0],shrink=0.85)
        rm=_radial_map(self.rows,self.cols); rf=rm.ravel(); re=np.linspace(0.0,rf.max(),31)
        rc2,rr,cr=[],[],[]
        for i in range(30):
            m=(rf>=re[i])&(rf<re[i+1])
            if m.sum()>=4: rc2.append((re[i]+re[i+1])/2); rr.append(np.median(T_raw.ravel()[m])); cr.append(np.median(T_corrected.ravel()[m]))
        axes[1,1].plot(rc2,rr,"r-o",ms=4,label="원본"); axes[1,1].plot(rc2,cr,"b-s",ms=4,label="보정후")
        axes[1,1].set_xlabel("반경 r"); axes[1,1].set_ylabel("온도 [C]"); axes[1,1].set_title("(d) 반경 프로파일")
        axes[1,1].legend(); axes[1,1].grid(alpha=0.3)
        fig.suptitle("열화상 광학 왜곡 보정 결과",fontsize=13,fontweight="bold"); plt.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(save_path)),exist_ok=True)
        plt.savefig(save_path,dpi=150,bbox_inches="tight"); plt.close(); return save_path

def add_synthetic_artifacts(T_clean,vignetting_strength=3.0,fpn_column_strength=0.8,
                             fpn_pixel_strength=0.3,narcissus_strength=1.5,seed=7):
    rows,cols=T_clean.shape; rng=np.random.RandomState(seed); T=T_clean.copy()
    rm=_radial_map(rows,cols)
    T+=-vignetting_strength*(rm**2)
    T+=rng.uniform(-fpn_column_strength,fpn_column_strength,cols)[np.newaxis,:]
    T+=rng.normal(0,fpn_pixel_strength,(rows,cols))
    T+=-narcissus_strength*np.exp(-((rm-0.7)**2)/(2*0.12**2))
    return T
