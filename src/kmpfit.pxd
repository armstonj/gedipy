"""
Copyright (c) 2010-2015, Kapteyn Astronomical Institute, University
of Groningen. All rights reserved.
Copyright (c) 2020, Modifications for use in GEDIPy

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following
      disclaimer in the documentation and/or other materials provided
      with the distribution.

    * Neither the name of the Kapteyn Astronomical Institute nor the names
      of its contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

# cython: language_level=3

cdef extern from "mpfit.h":

   ctypedef struct mp_par:
      int fixed
      int limited[2]
      double limits[2]
      char *parname
      double step
      double relstep
      int side
      int deriv_debug
      double deriv_reltol
      double deriv_abstol

   ctypedef struct mp_config:
      double ftol
      double xtol
      double gtol
      double epsfcn
      double stepfactor
      double covtol
      int maxiter
      int maxfev
      int douserscale
      int nofinitecheck

   ctypedef struct mp_result:
      double bestnorm
      double orignorm
      int niter
      int nfev
      int status
      int npar
      int nfree
      int npegged
      int nfunc
      double *resid
      double *xerror
      double *covar
      char version[20]

   ctypedef int (*mp_func)(int *m,
                       int n,
                       double *x,
                       double **fvec,
                       double **dvec,
                       void *private_data)

   cdef int mpfit(mp_func funct, int npar,
                 double *xall, mp_par *pars, mp_config *config,
                 void *private_data,
                 mp_result *result)

