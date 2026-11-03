"""Profile Gaussian working likelihood: oriented filament versus a planar background plus spot.

This is a model-dependent score, not a posterior probability or a calibrated physical-noise test.
"""
import numpy as np

SIDE=31
VARIANCE_FLOOR=1/12  # Explicit constrained-model floor in squared byte-value units.


def patch_at(image, x, y):
    # No resampling. Reflect padding is used only at the outer source-image border.
    r=SIDE//2;x=int(round(x));y=int(round(y))
    x=max(0,min(image.shape[1]-1,x));y=max(0,min(image.shape[0]-1,y))
    crop=image[max(0,y-r):min(image.shape[0],y+r+1),max(0,x-r):min(image.shape[1],x+r+1)]
    if crop.shape!=(SIDE,SIDE):crop=np.pad(crop,((max(0,r-y),max(0,y+r+1-image.shape[0])),(max(0,r-x),max(0,x+r+1-image.shape[1]))),mode='reflect')
    return crop.ravel().astype(np.float64)


class FilamentLikelihood:
    def __init__(self):
        yy,xx=np.mgrid[-15:16,-15:16];self.X=np.c_[np.ones(SIDE*SIDE),xx.ravel(),yy.ravel()]
        self.projection=np.linalg.pinv(self.X)
        self.line_parameters=[];line=[];blob=[];self.blob_parameters=[]
        # Shape grids are fixed before evaluation. Shared shapes are fitted on fit-image labels only.
        for width in (1.,2.,3.,4.):
            for length in (6.,10.,14.):
                if length/width<2.5:continue
                for angle in np.linspace(0,np.pi,12,endpoint=False):
                    u=xx*np.cos(angle)+yy*np.sin(angle);v=-xx*np.sin(angle)+yy*np.cos(angle)
                    line.append(np.exp(-.5*((u/length)**2+(v/width)**2)).ravel());self.line_parameters.append((width,length,float(angle)))
        for radius in (1.,2.,3.,4.,6.,8.):
            blob.append(np.exp(-.5*(xx**2+yy**2)/radius**2).ravel());self.blob_parameters.append(radius)
        self.line=self._normalise(np.asarray(line));self.blob=self._normalise(np.asarray(blob))
        self.selected_line=None;self.selected_blob=None

    def _normalise(self,templates):
        residual=templates-(self.X@(self.projection@templates.T)).T
        return residual/np.linalg.norm(residual,axis=1,keepdims=True)

    def residuals(self,patches):
        y=np.asarray(patches,np.float64)
        return y-(self.X@(self.projection@y.T)).T

    @staticmethod
    def log_likelihood(rss):
        variance=np.maximum(VARIANCE_FLOOR,np.asarray(rss)/(SIDE*SIDE))
        return -.5*((SIDE*SIDE)*np.log(2*np.pi*variance)+rss/variance)

    def rss(self,patches,templates):
        residual=self.residuals(patches);null=np.sum(residual**2,axis=1)
        amplitude=np.maximum(0,residual@templates.T)
        return np.maximum(0,null[:,None]-amplitude**2)

    def fit(self,positive_patches,negative_spot_patches):
        # Shared width/length: maximum total profile likelihood, with orientation optimised per positive.
        ll=self.log_likelihood(self.rss(positive_patches,self.line));groups={}
        for i,(w,l,angle) in enumerate(self.line_parameters):groups.setdefault((w,l),[]).append(i)
        scores={key:float(ll[:,indices].max(axis=1).sum()) for key,indices in groups.items()}
        shape=max(scores,key=scores.get);self.selected_line=groups[shape]
        spot_ll=self.log_likelihood(self.rss(negative_spot_patches,self.blob)).sum(axis=0)
        self.selected_blob=int(np.argmax(spot_ll))
        return {'line_width_pixels':shape[0],'line_length_scale_pixels':shape[1],
                'null_spot_radius_pixels':self.blob_parameters[self.selected_blob],
                'patch_size':SIDE,'variance_floor':VARIANCE_FLOOR,
                'line_grid_profile_log_likelihood':{str(k):v for k,v in scores.items()},
                'fit_positives':len(positive_patches),'fit_annotated_centrosomal_negatives':len(negative_spot_patches),
                'estimation':'Constrained Gaussian MLE: planar coefficients, nonnegative amplitude and residual variance; grid MLE for shared line shape and null spot radius; per-patch orientation profile MLE.',
                'interpretation':'Working log-likelihood ratio. No chi-square p-values or posterior probabilities are assigned.'}

    def score(self,patches):
        if self.selected_line is None:raise RuntimeError('Fit parameters on training source images first.')
        lrss=self.rss(patches,self.line[self.selected_line]).min(axis=1)
        brss=self.rss(patches,self.blob[[self.selected_blob]])[:,0]
        return self.log_likelihood(lrss)-self.log_likelihood(brss)


def threshold_at_fpr(negative_scores, alpha=.05):
    scores=np.asarray(negative_scores,float)
    if not len(scores) or not np.isfinite(scores).all():raise ValueError('Finite calibration-negative scores are required.')
    # Accept score >= threshold; choose a strict observed order statistic to handle ties conservatively.
    index=int(np.floor(alpha*len(scores)))
    return float(np.nextafter(np.sort(scores)[::-1][min(index,len(scores)-1)],np.inf))
