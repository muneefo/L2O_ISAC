"""Exact 1D ordered-spacing projection, wavelength coordinates and derivatives."""
import math
import numpy as np
import torch


def check_geometry(n,aperture,min_spacing):
    if n<1 or min_spacing<0 or aperture<(n-1)*min_spacing:
        raise ValueError('Aperture cannot accommodate antenna count/minimum spacing')


def _pav_row(y,upper):
    blocks=[]
    for i,value in enumerate(y):
        blocks.append([float(value),1,i,i+1])
        while len(blocks)>1 and blocks[-2][0]>blocks[-1][0]:
            right=blocks.pop();left=blocks.pop();count=left[1]+right[1]
            blocks.append([(left[0]*left[1]+right[0]*right[1])/count,count,left[2],right[3]])
    result=np.empty_like(y);labels=np.empty(len(y),dtype=np.int64);free=np.empty(len(y),dtype=bool)
    for k,(value,count,start,end) in enumerate(blocks):
        result[start:end]=np.clip(value,0,upper);labels[start:end]=k;free[start:end]=(0<value<upper)
    return result,labels,free


class _Projection(torch.autograd.Function):
    @staticmethod
    def forward(ctx,proposal,aperture,min_spacing):
        n=proposal.shape[-1];check_geometry(n,aperture,min_spacing)
        offset=torch.arange(n,dtype=proposal.dtype,device=proposal.device)*min_spacing
        shifted=(proposal-offset).detach().cpu().numpy().reshape(-1,n)
        projected=[];labels=[];free=[]
        for row in shifted:
            p,l,f=_pav_row(row,aperture-(n-1)*min_spacing);projected.append(p);labels.append(l);free.append(f)
        label=torch.as_tensor(np.array(labels),device=proposal.device);active=torch.as_tensor(np.array(free),device=proposal.device)
        ctx.save_for_backward(label,active);ctx.shape=proposal.shape
        return torch.as_tensor(np.array(projected),dtype=proposal.dtype,device=proposal.device).reshape(proposal.shape)+offset

    @staticmethod
    def backward(ctx,grad_output):
        labels,free=ctx.saved_tensors;g=grad_output.reshape(labels.shape);outputs=[]
        # Piecewise-exact Jacobian: mean on free pooled blocks; zero on clamped.
        # Torch operations retain differentiation in upstream gradients; no STE.
        for row,label,mask in zip(g,labels,free):
            values=[]
            for i in range(len(row)):
                block=label==label[i];values.append(row[block].mean() if mask[i] else row[i]*0)
            outputs.append(torch.stack(values))
        return torch.stack(outputs).reshape(ctx.shape),None,None


def project_positions(proposal,aperture,min_spacing):
    return _Projection.apply(proposal,float(aperture),float(min_spacing))


def steering(theta,x):
    """theta radians [...,paths], x in wavelengths [...,Nt]."""
    return torch.exp(-2j*math.pi*theta.sin()[...,None]*x[...,None,:])


def steering_derivative(theta,x):
    return -2j*math.pi*theta.sin()[...,None]*steering(theta,x)


def physical_steering(theta,p_meters,wavelength_m):
    if wavelength_m<=0:raise ValueError('Positive wavelength required')
    return steering(theta,p_meters/wavelength_m)


def geometry_residuals(x,aperture,min_spacing):
    return {'left_violation':(-x[...,0]).clamp_min(0),'right_violation':(x[...,-1]-aperture).clamp_min(0),
            'spacing_violation':(min_spacing-(x[...,1:]-x[...,:-1])).clamp_min(0).amax(-1) if x.shape[-1]>1 else torch.zeros_like(x[...,0])}
