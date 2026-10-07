"""Trabalho Pratico 2: ICA cega em quatro imagens 128x128.
Uso: python experimento.py --input ../mist_images.mat
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import scipy.io as sio
from scipy.stats import kurtosis
from scipy.optimize import minimize
from sklearn.decomposition import FastICA
from sklearn.feature_selection import mutual_info_regression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
FIG = BASE/'figuras'
FIG.mkdir(exist_ok=True)

def normalize_display(x):
    lo,hi = np.percentile(x,[1,99])
    if hi-lo < 1e-12: return np.zeros_like(x)
    return np.clip((x-lo)/(hi-lo),0,1)

def display_images(signals, name, title):
    fig,axs=plt.subplots(1,signals.shape[1],figsize=(13,3.35),constrained_layout=True)
    for i,ax in enumerate(axs):
        # Dados MATLAB: vetor linearizado por coluna (ordem F)
        a=signals[:,i].reshape(128,128,order='F')
        ax.imshow(normalize_display(a),cmap='gray',vmin=0,vmax=1)
        ax.set_title(f'Componente {i+1}')
        ax.axis('off')
    fig.suptitle(title)
    fig.savefig(FIG/f'{name}.png',dpi=165)
    plt.close(fig)

def histograms(signals,name,title):
    fig,axs=plt.subplots(1,signals.shape[1],figsize=(13,3),constrained_layout=True)
    for i,ax in enumerate(axs):
        ax.hist(signals[:,i],bins=65,density=True)
        ax.set_title(f'Sinal {i+1}')
    fig.suptitle(title)
    fig.savefig(FIG/f'{name}.png',dpi=145)
    plt.close(fig)

def spectra(signals,name,title):
    fig,axs=plt.subplots(1,signals.shape[1],figsize=(13,3.35),constrained_layout=True)
    for i,ax in enumerate(axs):
        a=signals[:,i].reshape(128,128,order='F')
        ft=np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(a-a.mean()))))
        ax.imshow(ft,cmap='magma');ax.axis('off');ax.set_title(f'Espectro {i+1}')
    fig.suptitle(title);fig.savefig(FIG/f'{name}.png',dpi=150);plt.close(fig)

def correlations(signals,name):
    c=np.corrcoef(signals.T)
    fig,ax=plt.subplots(figsize=(4.5,4))
    im=ax.imshow(c,vmin=-1,vmax=1,cmap='coolwarm')
    for i in range(c.shape[0]):
        for j in range(c.shape[1]): ax.text(j,i,f'{c[i,j]:.2f}',ha='center',va='center',fontsize=10)
    ax.set_xticks(range(4),range(1,5));ax.set_yticks(range(4),range(1,5));fig.colorbar(im,ax=ax)
    fig.tight_layout();fig.savefig(FIG/f'{name}.png',dpi=155);plt.close(fig)
    return c

def jacobi_jade(x):
    """ICA por diagonalizacao conjunta de matrizes de cumulantes de ordem 4.
    Whitening Z = (X - mean) @ V, cov(Z) = I.
    Otimizacao ortogonal global das 6 rotacoes (n=4), com multiplos inicios.
    """
    n,p=x.shape
    mu=x.mean(axis=0);centered=x-mu
    eigval,eigvec=np.linalg.eigh(centered.T@centered/n)
    order=np.argsort(eigval)[::-1];eigval=eigval[order];eigvec=eigvec[:,order]
    if np.min(eigval) <= 1e-12*np.max(eigval): raise ValueError('Matriz de mistura com posto insuficiente')
    V=eigvec@np.diag(1/np.sqrt(eigval))
    z=centered@V
    cumul=[]
    eye=np.eye(p)
    for i in range(p):
        for j in range(i,p):
            # cum(z_a,z_b,z_i,z_j) = E[z_a z_b z_i z_j] - delta_ab delta_ij - delta_ai delta_bj - delta_aj delta_bi
            m=(z.T*(z[:,i]*z[:,j]))@z/n
            m-=eye*(1 if i==j else 0)
            m-=np.outer(eye[:,i],eye[:,j])+np.outer(eye[:,j],eye[:,i])
            cumul.append(m)
    cumul=np.stack(cumul)
    pairs=[(i,j) for i in range(p) for j in range(i+1,p)]
    def rotation(angles):
        Q=np.eye(p)
        for a,(i,j) in zip(angles,pairs):
            r=np.eye(p);c,s=np.cos(a),np.sin(a)
            r[i,i]=c;r[j,j]=c;r[i,j]=-s;r[j,i]=s
            Q=Q@r
        return Q
    def loss(angles):
        Q=rotation(angles)
        d=np.einsum('ai,kab,bj->kij',Q,cumul,Q,optimize=True)
        diag=np.diagonal(d,axis1=1,axis2=2)
        return float(np.sum(d*d)-np.sum(diag*diag))
    best=None
    rng=np.random.default_rng(24)
    for init in [np.zeros(len(pairs))]+[rng.uniform(-1.3,1.3,len(pairs)) for _ in range(3)]:
        res=minimize(loss,init,method='BFGS',options={'maxiter':350,'gtol':1e-8})
        if best is None or res.fun < best.fun:best=res
    Q=rotation(best.x)
    S=z@Q
    A=np.linalg.pinv(V@Q)
    reconstructed=S@A+mu
    return S,reconstructed,{'joint_diagonalization_loss':float(best.fun),'converged':bool(best.success)}

def pairwise_mi(s,seed=42):
    """MI kNN, estimador nao negativo; media dos dois sentidos."""
    rng=np.random.default_rng(seed)
    ids=rng.choice(len(s),size=min(5000,len(s)),replace=False)
    y=s[ids]
    vals=[]
    for i in range(y.shape[1]):
        for j in range(i+1,y.shape[1]):
            a=mutual_info_regression(y[:,[i]],y[:,j],n_neighbors=5,random_state=seed)[0]
            b=mutual_info_regression(y[:,[j]],y[:,i],n_neighbors=5,random_state=seed)[0]
            vals.append((a+b)/2)
    return float(np.mean(vals)),[float(v) for v in vals]

def metrics(x,s,reconstructed):
    c=np.corrcoef(s.T)
    off=np.abs(c[np.triu_indices(s.shape[1],1)])
    mi,mi_pairs=pairwise_mi(s)
    err=np.linalg.norm(x-reconstructed)/np.linalg.norm(x)
    return {'mean_abs_correlation':float(np.mean(off)),'max_abs_correlation':float(np.max(off)),
            'mean_pairwise_mi_nats':mi,'pairwise_mi_nats':mi_pairs,
            'excess_kurtosis':[float(v) for v in kurtosis(s,axis=0,fisher=True,bias=False)],
            'relative_reconstruction_error':float(err)}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',default=str(BASE.parent/'mist_images.mat'))
    args=parser.parse_args()
    data=sio.loadmat(args.input)
    x=np.asarray(data['X'],dtype=np.float64)
    if x.shape==(4,16384):x=x.T
    if x.shape!=(16384,4):raise ValueError(f'Esperado (16384,4), recebido {x.shape}')
    display_images(x,'misturas','Imagens misturadas observadas')
    spectra(x,'espectros_misturas','Espectros espaciais 2D das misturas')
    histograms(x,'hist_misturas','Histogramas das misturas')
    correlations(x,'correlacao_misturas')
    results={'shape':list(x.shape),'observed_mean':x.mean(axis=0).tolist(),
             'observed_std':x.std(axis=0).tolist(),'observed_min':x.min(axis=0).tolist(),
             'observed_max':x.max(axis=0).tolist(),'mix_metrics':metrics(x,x,x)}
    for label in ('fastica','jade'):
        t=time.perf_counter()
        if label=='fastica':
            algo=FastICA(n_components=4,algorithm='parallel',whiten='unit-variance',fun='logcosh',
                         max_iter=2000,tol=1e-6,random_state=42)
            s=algo.fit_transform(x)
            rec=algo.inverse_transform(s)
            other={'iterations':int(algo.n_iter_)}
        else:s,rec,other=jacobi_jade(x)
        duration=time.perf_counter()-t
        # Ambiguidade de sinal e escala: inverter/normalizar apenas para exibicao.
        display_images(s,f'componentes_{label}',f'Componentes estimadas — {label.upper()}')
        histograms(s,f'hist_{label}',f'Distribuicoes — {label.upper()}')
        spectra(s,f'espectros_{label}',f'Espectros espaciais 2D — {label.upper()}')
        correlations(s,f'correlacao_{label}')
        display_images(rec,f'reconstrucao_{label}',f'Misturas reconstruidas — {label.upper()}')
        np.savez_compressed(BASE/f'saida_{label}.npz',sources=s,reconstruction=rec)
        results[label]={**metrics(x,s,rec),'runtime_seconds':duration,**other}
    (BASE/'resultados.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(results,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
