from numba import float32, int32, float64
from numba import cuda,jit,njit,prange
import numpy as np
from random import randint as rd
"""
I'm Anodite , I will be the first most intelligent software in the history .
Anodite is AI based in Genetic algorithm with a few number of steps and huge amount of mutations
plus intelligent branching and linking algorithm
def randombetweenrange():
        r1=rdranges[0]
        r2=rdranges[1]
        k=r1-1.
        for t in range(1000000):
            k+=1.
            randall[0,0]=k
            if k==r2:
                k=r1-1.
    def randombetweenrange2():
        r1=rdranges[0]
        r2=rdranges[1]
        k=r1-1.
        for t in range(1000000):
            k+=1.
            randall[1,0]=k
            if k==r2:
                k=r1-1.

    def randomindx1():
        r1=0.
        r2=len(inputs[0])
        k=r1-1.
        for t in range(1000000):
            k+=1.
            randall[2,0]=k
            if k==r2-1.:
                k=r1-1.
    def randomindx2():
        r1=0.
        r2=len(outputs[0])
        k=r1-1.
        for t in range(10000000):
            k+=1
            randall[3,0]=k
            if k==r2-1.:
                k=r1-1.
    def tick1():
        k=0.
        for t in range(1000000):
            if k==0.:
                k=1.
            else:
                k=0.
            randall[4,0]=k
    def tick2():
        k=0.
        for t in range(1000000):
            if k==0.:
                k=1.
            else:
                k=0.
            randall[5,0]=k
    def numofmutats():
        r1=1.
        r2=maxmutats
        k=r1-1.
        for t in range(1000000):
            k+=1.
            randall[6,0]=k
            if k==r2:
                k=r1-1.
    def rand5():
        r1=0.
        r2=5.
        k=r1-1
        for t in range(1000000):
            k+=1
            randall[7,0]=k
            if k==r2:
                k=r1-1.
    def randominputs():
        ans=0
        if len(inputs[0])>1:
            return rd(0,len(inputs[0])-1)
        return ans
    def randomoutputs():
        ans=0
        if len(outputs[0])>1:
            return rd(0,len(outputs[0])-1)
        return ans
 def calc():
      ll=l
      no=np.copy(outputs)
      mutatneuron()
      this_neuron=mainneuron[0]
      listrewards[ll,0]=0.
      mn=allneurons[ll]
      type=0
      indx=0
      itrs=0
      rwrd=0.
      ds=0#doyesno[0,0]
      r=int(ll+0.)
      #g[0]=ndones[0,0]
      #0.          0.         -5.58511111  0.          0.
      if True:
        this_neuron=mainneuron[0]
        mn=allneurons[ll]
        type=0
        indx=0
        itrs=0
        rwrd=0.
        ans=1.
        for k in range(len(inputs)):
           type=0
           indx=0
           itrs=0
           minans=0.
           for i in range(len(outputs[0])):
               no[k][i]=.00001
           for tver in range(100000000000000000):
            prep=0.
            if indx==position:
                this_neuron=mn
            for this_neuronu in this_neuron:
             if this_neuronu[0]==0.:
                prep=inputs[k][int(this_neuronu[1])]
             else:
                prep=outputs[k][int(this_neuronu[1])]
             ans=.00001
             if type==1:
               if this_neuronu[4]==0.:
                  ans+=prep*this_neuronu[2]
               elif this_neuronu[4]==1.:
                  ans+=prep*this_neuronu[2]
             elif this_neuronu[4]==0.:
                no[k][int(this_neuronu[3])]+=prep*this_neuronu[2]
             else:
                no[k][int(this_neuronu[3])]+=prep*this_neuronu[2]
            itrs+=1
            no[k][int(this_neuronu[3])]=np.tanh(no[k][int(this_neuronu[3])])
            #ans=np.tanh(ans)
            if itrs>=maxloops:
                break;
            if type==0:
                indx=mainshape[indx][1]
            else:
                if ans>=0:
                 indx=mainshape[indx][1]
                else:
                 indx=mainshape[indx][2]
            if indx==-1:
                break
            type=mainshape[indx][0]
           rw=rewarder(no[k][0],k)
           listrewards[ll,0]+=rw
"""
from tkinter import Tk
from tkinter import ttk
#@jit(parallel=True)
#@jit(parallel=True)
def cudacalcmutat(rdranges,n_oflists,steps,n_n,listallneurons,listoldlen,best1,maxmutats,mainneuron,mainshape,position,inputs,outputs,settingsofmainn,g,q,randall,allneurons,listbackup,listrewards,listarranges,layers,position1,position2,islayered,lnetwork,dnetwork,layerundertrain,over,gameid):
  for l in prange(n_oflists):
    maxloops=settingsofmainn[0]
    ll=0
    def mutatneuron():
        ll=l
        this_neuron=allneurons[ll]
        nt=(( (ll-listoldlen)*listoldlen/ (len(allneurons)-listoldlen) ))
        #[(0,1),(0,len-1),(100.0,-100.0),0 to len-1,(0,1)]
        if ll>listoldlen:
                old_neuron=allneurons[int(nt)]
                mutats=maxmutats#rd(0,maxmutats)
                if over!=-1:
                    mutats=1
              #  if rd(0,3)==0:
                 #   mutats=1
                for ii in range(len(old_neuron)):
                 for ii2 in range(len(old_neuron[ii])):
                    this_neuron[ii][ii2]=old_neuron[ii][ii2]+0.
                for i in range(int(mutats)):
                    which1=rd(0,len(this_neuron)-1)
                    which=rd(0,len(this_neuron[0])-1)
                    if mutats==len(this_neuron) or i<len(this_neuron):
                        which1=i
                    if over!=-1:
                        which1=over
                    which=2
                    which=int(which)
                   # if which1<len(inputs[0]):
                    #    this_neuron[which1][1]=which1
                   # if which==0 :
                    #    this_neuron[which1][which]=rd(0,1)
                     #   if this_neuron[which1][0]==0.:

                      #          this_neuron[which1][1]=randominputs()
                       # else:
                        #     this_neuron[which1][1]=randomoutputs()
                   # elif which==1 or which==3:
                    #    if this_neuron[which1][0]==0. and which !=3:

                     #           this_neuron[which1][1]=randominputs()
                      #  else:
                       #      this_neuron[which1][which]=randomoutputs()
                    #elif which==4:
                     #   this_neuron[which1][which]=rd(0,1)
                    if which==2 :
                        re=rd(0,2)
                        ####3
                        q=0
                        if re==0:
                            q=rd(90000,110000)/100000.
                        elif re==1:
                            q=rd(95000,105000)/100000.
                        elif re==2 :
                            q=rd(50000,200000)/100000.
                        if rd(0,maxmutats)*rd(0,1)==0:
                            q*=-1
                        this_neuron[which1][which]*=q
                        if this_neuron[which1][which]==0.:
                            this_neuron[which1][which]=0.00001

    def rewarder(no2,t):
        q=0.
        qinput=inputs[t]
        if gameid==0:
         real=np.tanh(inputs[t][0]*0.5+inputs[t][1]*-0.2+inputs[t][2]*0.6+inputs[t][3]*0.9+inputs[t][4]*0.1)#+inputs[t][5]*0.63+inputs[t][6]*-0.42+inputs[t][7]*0.45+inputs[t][8]*-0.25+inputs[t][9]*0.05+inputs[t][10]*0.35+inputs[t][11]*-0.7+inputs[t][12]*0.64+inputs[t][13]*0.78+inputs[t][14]*0.07+inputs[t][15]*0.3+inputs[t][16]*0.32+inputs[t][17]*0.455+inputs[t][18]*0.29+inputs[t][19]*0.19)
         vr=((no2-real)/real)
         if vr<=0:
             vr*=-1
         q=np.tanh(1-vr)
        elif gameid==1:
            newposition=0
            if inputs[t][3]==-1.:
                newposition=(0 if no2<0. else 1)
            elif inputs[t][3]<1.:
                if no2>=-0.5 and no2<=0.5:
                    if inputs[t][3]==-1.:
                        newposition=0
                    elif inputs[t][3]==1.:
                        newposition=2
                    else:
                        newposition=1
                else:
                    if no2>0:
                        newposition=2
                    else:
                        newposition=0
            if inputs[t][3]==1.:
                newposition=(1 if no2<0. else 2)
            if inputs[t][newposition]==1.:
                q=-1.
            else:
                q=1.
        return q**1
   
           #[]
    def lcalc():
      ll=l
      no=np.copy(outputs)
      mutatneuron()
      this_neuron=mainneuron[0]
      listrewards[ll,0]=0.
      mn=allneurons[ll]
      type=0
      indx=0
      itrs=0
      rwrd=0.
      ds=0#doyesno[0,0]
      r=int(ll+0.)
      #g[0]=ndones[0,0]
      #0.          0.         -5.58511111  0.          0.
      #layers,positon1,position2,islayered
      if True:
        this_neuron=mainneuron[0]
        mn=allneurons[ll]
        if islayered :
            if position1!=-1:
                layers[ll][position1][position2]=mn
        type=0
        indx=0
        itrs=0
        rwrd=0.
        ans=1.
        for i in range(len(inputs)):
            tinput=np.copy(inputs[i])
            for k in range(len(outputs[0])):
               no[i][k]=0.00000000001
            indx=0
            itrs=0
            for iff in range(19090290481390):
                if islayered :
                    if layerundertrain and indx==position:
                        layer1=position1
                        layer2=0
                        ttinputs=np.copy(tinput)
                        while layer1<len(lnetwork[indx]) :#and layer2<=position2:
                             c=lnetwork[indx][layer1][layer2]
                             if layer1==position1 and layer2==position2:
                                c=mn#layers[ll][position1][position2]
                             tinput[layer2]=0.000000001
                             tr=0
                             for cu in c:
                                 tinput[layer2]+= ttinputs[tr]*cu[2]
                                 tr+=1
                             tinput[layer2]=np.tanh(tinput[layer2])##
                             layer2+=1
                             f=len(lnetwork[indx][layer1])
                             if layer2 ==len(lnetwork[indx][layer1]) or (layer1==position1 and layer2-1==position2):
                                 ttinputs=np.copy(tinput)
                                 layer1+=1
                                 layer2=0
                    else:
                     for layer1 in range(len(lnetwork[indx])):
                        ttinputs=np.copy(tinput)
                        for layer2 in range(len(lnetwork[indx][layer1])):
                            c=lnetwork[indx][layer1][layer2]
                          #  if layer1==position1 and layer2==position2:
                         #       c=layers[ll][position1][position2]
                            tinput[layer2]=0.000000001
                            tr=0
                            for cu in c:
                                 tinput[layer2]+=ttinputs[tr]*cu[2]
                                 tr+=1
                            tinput[layer2]=np.tanh(tinput[layer2])##
                ans=0.
                for il in range(len(dnetwork[indx])):

                  c=dnetwork[indx][il]
                  if indx==position and not islayered:
                      c=mn
                  if type==0:
                    tr=0
                    for cu in c:
                        an= tinput[tr]*cu[2]
                        no[i][il]+=an
                        tr+=1
                    no[i][il]=np.tanh(no[i][il])
                  elif type==1:
                    tr=0
                    for cu in c:
                        ans+= tinput[tr]*cu[2]
                        tr+=1
                    ans=np.tanh(ans)
                itrs+=1
                if itrs>=maxloops:
                   break;
                if type==0:
                   indx=mainshape[indx][1]
                else:
                  if ans>=0:
                     indx=mainshape[indx][1]
                  else:
                     indx=mainshape[indx][2]
                if indx==-1:
                   break
                type=mainshape[indx][0]
            rw=rewarder(no[i][0],i)
            listrewards[ll,0]+=rw
    if l>=0:
        lcalc()
def cudacalcmutat2(rdranges,n_oflists,steps,n_n,listallneurons,listoldlen,best1,maxmutats,mainneuron,mainshape,position,inputs,outputs,settingsofmainn,g,q,randall,allneurons,listbackup,listrewards,listarranges,layers,position1,position2,islayered,lnetwork,dnetwork,layerundertrain,over,gameid):
  for l in prange(n_oflists):
    maxloops=settingsofmainn[0]
    ll=0
    def mutatneuron():
        ll=l
        this_neuron=allneurons[ll]
        nt=(( (ll-listoldlen)*listoldlen/ (len(allneurons)-listoldlen) ))
        #[(0,1),(0,len-1),(100.0,-100.0),0 to len-1,(0,1)]
        if ll>listoldlen:
                old_neuron=allneurons[int(nt)]
                mutats=maxmutats#rd(0,maxmutats)
                if over!=-1:
                    mutats=1
              #  if rd(0,3)==0:
                 #   mutats=1
                for ii in range(len(old_neuron)):
                 for ii2 in range(len(old_neuron[ii])):
                    this_neuron[ii][ii2]=old_neuron[ii][ii2]+0.
                for i in range(int(mutats)):
                    which1=rd(0,len(this_neuron)-1)
                    which=rd(0,len(this_neuron[0])-1)
                    if mutats==len(this_neuron) or i<len(this_neuron):
                        which1=i
                    if over!=-1:
                        which1=over
                    which=2
                    which=int(which)
                   # if which1<len(inputs[0]):
                    #    this_neuron[which1][1]=which1
                   # if which==0 :
                    #    this_neuron[which1][which]=rd(0,1)
                     #   if this_neuron[which1][0]==0.:

                      #          this_neuron[which1][1]=randominputs()
                       # else:
                        #     this_neuron[which1][1]=randomoutputs()
                   # elif which==1 or which==3:
                    #    if this_neuron[which1][0]==0. and which !=3:

                     #           this_neuron[which1][1]=randominputs()
                      #  else:
                       #      this_neuron[which1][which]=randomoutputs()
                    #elif which==4:
                     #   this_neuron[which1][which]=rd(0,1)
                    if which==2 :
                        re=rd(0,2)
                        ####3
                        q=0
                        if re==0:
                            q=rd(90000,110000)/100000.
                        elif re==1:
                            q=rd(95000,105000)/100000.
                        elif re==2 :
                            q=rd(50000,200000)/100000.
                        if rd(0,maxmutats)*rd(0,1)==0:
                            q*=-1
                        this_neuron[which1][which]*=q
                        if this_neuron[which1][which]==0.:
                            this_neuron[which1][which]=0.00001

    def rewarder(no2,t):
        q=0.
        qinput=inputs[t]
        if gameid==0:
         real=np.tanh(inputs[t][0]*0.5+inputs[t][1]*-0.2+inputs[t][2]*0.6+inputs[t][3]*0.9+inputs[t][4]*0.1)#+inputs[t][5]*0.63+inputs[t][6]*-0.42+inputs[t][7]*0.45+inputs[t][8]*-0.25+inputs[t][9]*0.05+inputs[t][10]*0.35+inputs[t][11]*-0.7+inputs[t][12]*0.64+inputs[t][13]*0.78+inputs[t][14]*0.07+inputs[t][15]*0.3+inputs[t][16]*0.32+inputs[t][17]*0.455+inputs[t][18]*0.29+inputs[t][19]*0.19)
         vr=((no2-real)/real)
         if vr<=0:
             vr*=-1
         q=np.tanh(1-vr)
        elif gameid==1:
            newposition=0
            if inputs[t][3]==-1.:
                newposition=(0 if no2<0. else 1)
            elif inputs[t][3]<1.:
                if no2>=-0.5 and no2<=0.5:
                    if inputs[t][3]==-1.:
                        newposition=0
                    elif inputs[t][3]==1.:
                        newposition=2
                    else:
                        newposition=1
                else:
                    if no2>0:
                        newposition=2
                    else:
                        newposition=0
            if inputs[t][3]==1.:
                newposition=(1 if no2<0. else 2)
            if inputs[t][newposition]==1.:
                q=-1.
            else:
                q=1.
        return q**1
   
           #[]
    def lcalc():
      ll=l
      no=np.copy(outputs)
      mutatneuron()
      this_neuron=mainneuron[0]
      listrewards[ll,0]=0.
      mn=allneurons[ll]
      type=0
      indx=0
      itrs=0
      rwrd=0.
      ds=0#doyesno[0,0]
      r=int(ll+0.)
      #g[0]=ndones[0,0]
      #0.          0.         -5.58511111  0.          0.
      #layers,positon1,position2,islayered
      if True:
        this_neuron=mainneuron[0]
        mn=allneurons[ll]
        if islayered :
            if position1!=-1:
                layers[ll][position1][position2]=mn
        type=0
        indx=0
        itrs=0
        rwrd=0.
        ans=1.
        for i in range(len(inputs)):
            tinput=np.copy(inputs[i])
            for k in range(len(outputs[0])):
               no[i][k]=0.00000000001
            indx=0
            itrs=0
            for iff in range(19090290481390):
                if islayered :
                    if layerundertrain and indx==position:
                        layer1=position1
                        layer2=0
                        ttinputs=np.copy(tinput)
                        while layer1<len(lnetwork[indx]) :#and layer2<=position2:
                             c=lnetwork[indx][layer1][layer2]
                             if layer1==position1 and layer2==position2:
                                c=mn#layers[ll][position1][position2]
                             tinput[layer2]=0.000000001
                             tr=0
                             for cu in c:
                                 tinput[layer2]+= ttinputs[tr]*cu[2]
                                 tr+=1
                             tinput[layer2]=np.tanh(tinput[layer2])##
                             layer2+=1
                             f=len(lnetwork[indx][layer1])
                             if layer2 ==len(lnetwork[indx][layer1]) or (layer1==position1 and layer2-1==position2):
                                 ttinputs=np.copy(tinput)
                                 layer1+=1
                                 layer2=0
                    else:
                     for layer1 in range(len(lnetwork[indx])):
                        ttinputs=np.copy(tinput)
                        for layer2 in range(len(lnetwork[indx][layer1])):
                            c=lnetwork[indx][layer1][layer2]
                          #  if layer1==position1 and layer2==position2:
                         #       c=layers[ll][position1][position2]
                            tinput[layer2]=0.000000001
                            tr=0
                            for cu in c:
                                 tinput[layer2]+=ttinputs[tr]*cu[2]
                                 tr+=1
                            tinput[layer2]=np.tanh(tinput[layer2])##
                ans=0.
                for il in range(len(dnetwork[indx])):

                  c=dnetwork[indx][il]
                  if indx==position and not islayered:
                      c=mn
                  if type==0:
                    tr=0
                    for cu in c:
                        an= tinput[tr]*cu[2]
                        no[i][il]+=an
                        tr+=1
                    no[i][il]=np.tanh(no[i][il])
                  elif type==1:
                    tr=0
                    for cu in c:
                        ans+= tinput[tr]*cu[2]
                        tr+=1
                    ans=np.tanh(ans)
                itrs+=1
                if itrs>=maxloops:
                   break;
                if type==0:
                   indx=mainshape[indx][1]
                else:
                  if ans>=0:
                     indx=mainshape[indx][1]
                  else:
                     indx=mainshape[indx][2]
                if indx==-1:
                   break
                type=mainshape[indx][0]
            rw=rewarder(no[i][0],i)
            listrewards[ll,0]+=rw
    if l>=0:
        lcalc()

###################
@jit
def arrange(listrewards,listarranges,rww):
        for rw in range(listrewards.shape[0]):
           rrw=listrewards[rw,0]
           if rrw>rww :#or rrw<=rww: 
            for rw2 in range(listrewards.shape[0]):
                if rrw>listrewards[rw2,0] and listarranges[rw2,0]<listarranges[rw,0]:
                    og=listarranges[rw,0]
                    listarranges[rw,0]=listarranges[rw2,0]
                    rg=listarranges[rw2,0]
                    for l in range(listarranges.shape[0]):
                        if listarranges[l,0]>=rg and listarranges[l]<og and l!=rw:
                            listarranges[l,0]+=1
@jit
def manage(rw,listrewards,listbackup,listarranges,allneurons):
        arrange(listrewards,listarranges,rw[0])
        for t in range(len(listrewards)):
             gt=listarranges[t][0]
             listbackup[gt]=allneurons[t]
            # llistrrr[gt]=listrewards[t]
             if listrewards[t][0]>=rw[0]:
                 rw[0]=listrewards[t][0]+0.
import time
def getinputsoutputs(n,l):
    inputs=np.array([[0.]*l]*n);outputs=np.array([[0.]]*n);
    for t in range(n):
        for j in range(l):
            inputs[t][j]=rd(-99999,99999)/100000.
    for t in range(n):
            outputs[t][0]=rd(-99999,99999)/100000.
    return inputs,outputs
def getinputsoutputs3(n,l):
    inputs=np.array([[0.001]*l]*n);outputs=np.array([[1.]]*n);
    for t in range(n):
            q=rd(0,len(inputs[0])-3)
            q1=q
            inputs[t][q]=1.
            inputs[t][3]=rd(-1,1)+0.
            if inputs[t][3]==0:
                inputs[t][3]=0.001
            q=rd(0,len(inputs[0])-3)
            inputs[t][q]=1.
            inputs[t][3]=rd(-1,1)+0.
            if inputs[t][3]==0 or abs(q1-q)==1:
                inputs[t][3]=0.001
            inputs[t][-1]=1.

    return inputs,outputs
def getinputsoutputs2(n,l):
    inputs=np.array([[0.]*l]*n);outputs=np.array([[0.]]*n);
    for t in range(n):
        for j in range(l):
            inputs[t][j]=rd(-99999,99999)/100000.
    for t in range(n):
            outputs[t][0]=np.tanh(inputs[t][0]*0.5+inputs[t][1]*-0.2+inputs[t][2]*0.6+inputs[t][3]*0.9)#+inputs[t][4]*0.1)#+inputs[t][5]*0.63+inputs[t][6]*-0.42+inputs[t][7]*0.45+inputs[t][8]*-0.25+inputs[t][9]*0.05)#+inputs[t][10]*0.35+inputs[t][11]*-0.7+inputs[t][12]*0.64+inputs[t][13]*0.78+inputs[t][14]*0.07+inputs[t][15]*0.3+inputs[t][16]*0.32+inputs[t][17]*0.455+inputs[t][18]*0.29+inputs[t][19]*0.19)
    return inputs,outputs
def launch():
    n_n=20;
    inputs,outputs=getinputsoutputs2(10000,n_n);
    num_randoms=1;lenrands=100
    #for t in range(len(inputs)):
     #   print(outputs[t],inputs[t])
    n_oflists=n_n*10;listoldlen=10;steps=3500000;stoptime=5.8;
    rdranges=np.array([-100.,100.]);
    listallneurons=np.array([[[1.]*5]*n_n]*n_oflists,dtype=float);
    best1=np.array([[[1.]*5]*n_n]);maxmutats=n_n*2;mainneuron=np.array([ [[1.]*5]*n_n ]);mainshape=np.array([[0,-1,0]]);
    position=0;settingsofmainn=np.array([50]);
    print(mainneuron.shape)
    g=np.array([0.])
    q=np.array([[0.]]*200)
    randall=np.array([[0.]]*50)#cuda.shared.array(shape=(50, 1), dtype=float32)
    allneurons=np.array([[[0,0,1,0,1]]*n_n]*n_oflists,dtype=float)#cuda.shared.array(shape=(100,5),dtype=float32)
    listbackup=np.array([[[0,0,1,0,1]]*n_n]*n_oflists,dtype=float)#cuda.shared.array(shape=(100,5),dtype=float32)
    listrewards=np.array([[0.]]*n_oflists)#cuda.shared.array(shape=(100,1),dtype=float32)
    listarranges=np.array([[1.]]*n_oflists,dtype=float)#cuda.shared.array(shape=(100,1),dtype=int32)
    listarranges=np.array([[0]]*n_oflists)
    for i in range(len(listarranges)):
        listarranges[i]=i
    ut=np.copy(listarranges)
    ut2=np.copy(listrewards)
    rw=np.array([-55555555555.])
    
    #cudacalcmutat(rdranges,n_oflists,steps,n_n,listallneurons,listoldlen,best1,maxmutats,mainneuron,mainshape,position,inputs,outputs,settingsofmainn,g,q,randall,allneurons,listbackup,listrewards,listarranges)
    cudacalcmutat(rdranges,n_oflists,steps,n_n,listallneurons,listoldlen,best1,maxmutats,mainneuron,mainshape,position,inputs,outputs,settingsofmainn,g,q,randall,allneurons,listbackup,listrewards,listarranges)
    listarranges=np.copy(ut)
        #listbackup=np.array0([[1,1,1,1,1]]*100,dtype=float)
    manage(rw,listrewards,listbackup,listarranges,allneurons)
    allneurons=np.copy(listbackup)
    listrewards=np.copy(ut2)
    # 0.0009679794311523438
    #
    listarranges=np.copy(ut)
    
    data=np.array(list(zip(inputs,outputs)))
    inpts=[]
    otpts=[]
    for t in range(num_randoms):
        np.random.shuffle(data)
        inpts.append(np.array(list(data[:lenrands,0]),dtype=float))
        otpts.append(np.array(list(data[:lenrands,1]),dtype=float))
    #inpts=np.array(list(inpts),dtype=float);
    #otpts=np.array(list(otpts),dtype=float);
    print("APP started . . .")
    rw=np.array([-55555555555.])
    scnd=time.time()
    for i in range(steps):
        tl=(rd(0,num_randoms-1) if num_randoms>1 else 0)
        inputs=inpts[tl]
        outputs=otpts[tl]
        listoldlen=rd(1,int(n_oflists*0.1))
        cudacalcmutat(rdranges,n_oflists,steps,n_n,listallneurons,listoldlen,best1,maxmutats,mainneuron,mainshape,position,inputs,outputs,settingsofmainn,g,q,randall,allneurons,listbackup,listrewards,listarranges)
        listarranges=np.copy(ut)
        #if rd(0,2)==0:
        #rw[0]=-55555555555.
        manage(rw,listrewards,listbackup,listarranges,allneurons)
        allneurons=np.copy(listbackup)
        #print(rw[0]/len(inputs),allneurons[0],"time in seconds:",(time.time()-scnd))
        if i>=100:
            []
        if (time.time()-scnd)>=stoptime:
            print(i)
            break
    print(rw[0]/len(inputs),allneurons[0],"time in seconds:",(time.time()-scnd))
@jit
def rewarder(no2):
        real=np.tanh(0.3*1)
        vr=((no2-real)/real)**2
        return np.tanh(1-vr)
# it's time to add layers . . .
#launch()
def key(elem):
    return -elem[0]
def trainbylayers(n_layers,n_n2):
    n_n=n_n2;gameid=1
    lneuron=np.array([[ [[0.,0.,1.,0.,1.]]*n_n ]*n_n]*n_layers)
    islayered=False;
    inputs,outputs=getinputsoutputs3(10000,n_n);
    num_randoms=100;lenrands=100
    #for t in range(len(inputs)):
     #   print(outputs[t],inputs[t])
    n_oflists=n_n*5;listoldlen=5;steps=100000;stoptime=.15;
    rdranges=np.array([-100.,100.]);
    listallneurons=np.array([[[1.]*5]*n_n]*n_oflists,dtype=float);
    best1=np.array([[[1.]*5]*n_n]);maxmutats=n_n*2;mainneuron=np.array([ [[1.]*5]*n_n ]);mainshape=np.array([[0,-1,0]]);
    position=0;settingsofmainn=np.array([50]);
    print(mainneuron.shape)
    g=np.array([0.])
    q=np.array([[0.]]*200)
    randall=np.array([[0.]]*50)#cuda.shared.array(shape=(50, 1), dtype=float32)
    allneurons=np.array([[[0,0,1,0,1]]*n_n]*n_oflists,dtype=float)#cuda.shared.array(shape=(100,5),dtype=float32)
    listbackup=np.array([[[0,0,1,0,1]]*n_n]*n_oflists,dtype=float)#cuda.shared.array(shape=(100,5),dtype=float32)
    listrewards=np.array([[0.]]*n_oflists)#cuda.shared.array(shape=(100,1),dtype=float32)
    listarranges=np.array([[1.]]*n_oflists,dtype=float)#cuda.shared.array(shape=(100,1),dtype=int32)
    listarranges=np.array([[0]]*n_oflists)
    for i in range(len(listarranges)):
        listarranges[i]=i
    ut=np.copy(listarranges)
    ut2=np.copy(listrewards)
    rw=np.array([-55555555555.])
    dnetwork=np.array([ [[[0.,0.,1.,0.,1.]]*n_n] ])
    lnetwork=np.array([ lneuron ])
    layers=np.array( [np.copy(lneuron)]*n_oflists )
    layerundertrain=False
    over=rd(0,4)
    #cudacalcmutat(rdranges,n_oflists,steps,n_n,listallneurons,listoldlen,best1,maxmutats,mainneuron,mainshape,position,inputs,outputs,settingsofmainn,g,q,randall,allneurons,listbackup,listrewards,listarranges)
    cudacalcmutat(rdranges,n_oflists,steps,n_n,listallneurons,listoldlen,best1,maxmutats,mainneuron,mainshape,position,inputs,outputs,settingsofmainn,g,q,randall,allneurons,listbackup,listrewards,listarranges,layers,0,0,islayered,lnetwork,dnetwork,layerundertrain,over,gameid)
    listarranges=np.copy(ut)
        #listbackup=np.array0([[1,1,1,1,1]]*100,dtype=float)
    manage(rw,listrewards,listbackup,listarranges,allneurons)
    allneurons=np.copy(listbackup)
    listrewards=np.copy(ut2)
    # 0.0009679794311523438
    #
    listarranges=np.copy(ut)
    
    data=np.array(list(zip(inputs,outputs)))
    inpts=[]
    otpts=[]
    for t in range(num_randoms):
        np.random.shuffle(data)
        inpts.append(np.array(list(data[:lenrands,0]),dtype=float))
        otpts.append(np.array(list(data[:lenrands,1]),dtype=float))
    #inpts=np.array(list(inpts),dtype=float);
    #otpts=np.array(list(otpts),dtype=float);
    for iig in range(1):
      print("APP started . . .")
      rw=np.array([-55555555555.])
      scnd=time.time()
      layerundertrain=False
      for i in range(steps):
        tl=(rd(0,num_randoms-1) if num_randoms>1 else 0)
        inputs=inpts[tl]
        outputs=otpts[tl]
        #listoldlen=rd(1,int(n_oflists*0.1))
        over=-1#rd(0,n_n-1)
        cudacalcmutat(rdranges,n_oflists,steps,n_n,listallneurons,listoldlen,best1,maxmutats,mainneuron,mainshape,position,inputs,outputs,settingsofmainn,g,q,randall,allneurons,listbackup,listrewards,listarranges,layers,0,0,islayered,lnetwork,dnetwork,layerundertrain,over,gameid)
        listarranges=np.copy(ut)
        #if rd(0,2)==0:
        rw[0]=-55555555555.
        listrrr=np.copy(listrewards)
        manage(rw,listrewards,listbackup,listarranges,allneurons)
       # gi=list(listrewards)
       # gi.sort(key=key)
        allneurons=np.copy(listbackup)
        #print(rw[0]/len(inputs),allneurons[0],"time in seconds:",(time.time()-scnd))
        if i>=100:
            []
        if (time.time()-scnd)>=stoptime:
            print(i)
            break
      dnetwork[0][0]=np.copy(allneurons[0])
      print(rw[0]/len(inputs),allneurons[0],"time in seconds:",(time.time()-scnd))
      layerundertrain=True
      islayered=True
      for position1 in range(n_layers-1,-1,-1):
       []
       print("layer ",position1)
       for position2 in range(n_n): 
        scnd=time.time()
        allneurons=np.array([[[0,0,1,0,1]]*n_n]*n_oflists,dtype=float)
        for i in range(steps):
          tl=(rd(0,num_randoms-1) if num_randoms>1 else 0)
          inputs=inpts[tl]
          outputs=otpts[tl]
        #listoldlen=rd(1,int(n_oflists*0.1))
          over=-1#rd(0,n_n-1)
          cudacalcmutat(rdranges,n_oflists,steps,n_n,listallneurons,listoldlen,best1,maxmutats,mainneuron,mainshape,position,inputs,outputs,settingsofmainn,g,q,randall,allneurons,listbackup,listrewards,listarranges,layers,position1,position2,islayered,lnetwork,dnetwork,layerundertrain,over,gameid)
          listarranges=np.copy(ut)
        #if rd(0,2)==0:
          rw[0]=-55555555555.
          listrrr=np.copy(listrewards)
          manage(rw,listrewards,listbackup,listarranges,allneurons)
       # gi=list(listrewards)
       # gi.sort(key=key)
          allneurons=np.copy(listbackup)
        #print(rw[0]/len(inputs),allneurons[0],"time in seconds:",(time.time()-scnd))
          if i>=100:
            []
          if (time.time()-scnd)>=stoptime:
            print(i)
            break
        lnetwork[0][position1][position2]=np.copy(allneurons[0])
        print(rw[0]/len(inputs),allneurons[0],"time in seconds:",(time.time()-scnd))
    import os
    f=open('network.txt', 'w')
    for u in dnetwork:
     for r in u:
        for t in r:
            f.write(str(t[2])+"\n")
    for u in lnetwork:
     for r in u:
        for tt in r:
          for t in tt:
            f.write(str(t[2])+"\n")
    f.close()
    cudacalcmutat2(rdranges,n_oflists,steps,n_n,listallneurons,listoldlen,best1,maxmutats,mainneuron,mainshape,position,inputs,outputs,settingsofmainn,g,q,randall,allneurons,listbackup,listrewards,listarranges,layers,0,4,islayered,lnetwork,dnetwork,layerundertrain,over,gameid)
         
    rw=np.array([-55555555555.])
    print(dnetwork)
    print(lnetwork)
    return dnetwork,lnetwork
    []
    
#متنساش لما ترجع تشوف سبب خراب البرنامج مع حساب الطبقات
#خلاص كدةة البرنامج تمام الطبقات شغالة مية مية و الريجريش شغال فاضل أبدأ أوسع الفكرة أكتر
#البرنامج جاهز للمرحلة الجديدة إحنا وصلنا لدقة 99.9 % في 16 ثانية
#محتاج أجرب فكرة جديدة
for i in range(2,0,-1):
    []
dnetwork,lnetwork=trainbylayers(3,5)
import os
f=open('network.txt', 'w')
for u in dnetwork:
    for r in u:
        for t in r:
            f.write(str(t[2])+"\n")
for u in lnetwork:
    for r in u:
        for tt in r:
          for t in tt:
            f.write(str(t[2])+"\n")
[]
#f.write("\r\n"+str(lnetwork))

#"""

#root = Tk()
#frm = ttk.Frame(root, padding=10)
#frm.grid()
#ttk.Label(frm, text="Hello World!").grid(column=0, row=0)
#ttk.Button(frm, text="Quit", command=root.destroy).grid(column=1, row=0)
#root.mainloop()
#"""