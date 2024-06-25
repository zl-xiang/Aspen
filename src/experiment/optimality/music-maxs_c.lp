w(X,Y,1,2) :- eq(X,Y,1).
eq(X,Y,1) :- X!=Y,track(X,RI,MI,P,N,NA,ACI,L,IDT),track(Y,RI1,MI1,P,N1,NA1,ACI1,L1,IDT1),eq(MI,MI1,_),not empty(P),sim(NA,NA1,S),S>=95.
w(X,Y,2,2) :- eq(X,Y,2).
eq(X,Y,2) :- X!=Y,track(X,RI,MI,P,N,NA,ACI,L,IDT),track(Y,RI1,MI1,P1,N1,NA1,ACI1,L1,IDT1),eq(ACI,ACI1,_),sim(NA,NA1,S),S>=95.
w(X,Y,3,2) :- eq(X,Y,3).
eq(X,Y,3) :- X!=Y,recording(X,NA,ACI,L,V),recording(Y,NA1,ACI1,L,V),eq(ACI,ACI1,_),sim(NA,NA1,S),S>=90,not empty(L),not empty(V).
w(X,Y,4,2) :- eq(X,Y,4).
eq(X,Y,4) :- X!=Y,medium(X,REI,P,F,TC),medium(Y,REI1,P,F1,TC1),eq(REI,REI1,_),not empty(P).
w(X,Y,5,2) :- eq(X,Y,5).
eq(X,Y,5) :- X!=Y,release(X,RGI,AC,NA,B,S,PA,LA,SC,Q),release(Y,RGI1,AC1,NA1,B,S1,PA1,LA1,SC1,Q1),not empty(B).
w(X,Y,6,2) :- eq(X,Y,6).
eq(X,Y,6) :- X!=Y,release_group(X,NA,AC,TY),release_group(Y,NA1,AC1,TY1),eq(X,RGI,_),eq(Y,RGI1,_),eq(REI,REI1,_),release(REI,RGI,AC2,NA2,B2,S2,PA2,LA2,SC2,Q2),release(REI1,RGI1,AC3,NA3,B3,S3,PA3,LA3,SC3,Q3),sim(NA,NA1,S),S>=90.
w(X,Y,7,2) :- eq(X,Y,7).
eq(X,Y,7) :- X!=Y,release_group(X,NA,ACI,TY),release_group(Y,NA1,ACI1,TY1),eq(ACI,ACI1,_),eq(X,RGI,_),eq(Y,RGI1,_),eq(REI,REI1,_),release(REI,RGI,AC2,NA2,B2,S2,PA2,LA2,SC2,Q2),release(REI1,RGI1,AC3,NA3,B3,S3,PA3,LA3,SC3,Q3).
w(X,Y,8,2) :- eq(X,Y,8).
eq(X,Y,8) :- X!=Y,artist(X,NA,SN,BY,BM,BD,EY,EM,ED,END,AT,G,ARI,C),artist(Y,NA1,SN1,BY1,BM1,BD1,EY1,EM1,ED1,END1,AT1,G1,ARI1,C1),eq(X,AI,_),eq(Y,AI1,_),artist_credit_name(ACI,P,AI,NA4),artist_credit_name(ACI1,P,AI1,NA5),not empty(P),eq(ACI,ACI1,_).
w(X,Y,9,2) :- eq(X,Y,9).
eq(X,Y,9) :- X!=Y,area(X,NA,ART,END),area(Y,NA1,ART,END1),not empty(ART),eq(X,ARI,_),eq(Y,ARI1,_),eq(PI,PI1,_),place(PI,NA2,PT,ADD,ARI,CD,END2),place(PI1,NA3,PT1,ADD1,ARI1,CD1,END3),sim(NA,NA1,S),S>=90.
w(X,Y,10,2) :- eq(X,Y,10).
eq(X,Y,10) :- X!=Y,label(X,NA,LT,ARI,END1),label(Y,NA1,LT,ARI1,END2),eq(ARI,ARI1,_),not empty(LT),sim(NA,NA1,S),S>=90.
w(X,Y,11,2) :- eq(X,Y,11).
eq(X,Y,11) :- X!=Y,place(X,NA,PT,ADD,ARI1,CD,END),place(Y,NA1,PT,ADD1,ARI2,CD1,END1),not empty(PT),eq(ARI1,ARI2,_),sim(NA,NA1,S),S>=90.
w(X,Y,12,1) :- eq(X,Y,12).
active(X,Y,12) :- X!=Y,artist(X,NA,SN,BY,BM,BD,EY,EM,ED,END,AT,G,ARI,C),artist(Y,NA1,SN1,BY1,BM1,BD1,EY1,EM1,ED1,END1,AT1,G1,ARI1,C1),eq(ARI,ARI1,_),sim(NA,NA1,S),S>=95.
w(X,Y,13,1) :- eq(X,Y,13).
active(X,Y,13) :- X!=Y,release(X,RGI,AC,NA,B,ST,PA,LA,SC,Q),not empty(ST),release(Y,RGI1,AC1,NA1,B1,ST,PA1,LA1,SC1,Q1),eq(RGI,RGI1,_),sim(NA,NA1,S),S>=90.
w(X,Y,14,1) :- eq(X,Y,14).
active(X,Y,14) :- X!=Y,release_group(X,NA2,ACI,TY),release_group(Y,NA3,ACI1,TY1),eq(ACI,ACI1,_),sim(NA2,NA3,S),S>=90.
w(X,Y,15,1) :- eq(X,Y,15).
active(X,Y,15) :- X!=Y,artist(X,NA,SN,BY,BM,BD,EY,EM,ED,END,AT,G,ARI,C),not empty(BY),not empty(AT),artist(Y,NA1,SN1,BY,BM1,BD1,EY1,EM1,ED1,END1,AT,G1,ARI1,C1),sim(NA,NA1,S),S>=90.
w(X,Y,16,1) :- eq(X,Y,16).
active(X,Y,16) :- X!=Y,artist(X,NA,SN,BY,BM,BD,EY,EM,ED,END,AT,G,ARI,C),not empty(G),artist(Y,NA1,SN1,BY1,BM1,BD1,EY1,EM1,ED1,END1,AT1,G,ARI1,C1),sim(NA,NA1,S),S>=90,sim(SN,SN1,S1),S1>=90.
w(X,Y,17,1) :- eq(X,Y,17).
active(X,Y,17) :- X!=Y,track(X,RI,MI,P,N,NA,ACI,L,IDT),track(Y,RI1,MI1,P,N,NA1,ACI1,L,IDT),not empty(P),not empty(N),not empty(L),not empty(IDT),sim(NA,NA1,S),S>=85.
w(X,Y,18,1) :- eq(X,Y,18).
active(X,Y,18) :- X!=Y,recording(X,NA,ACI,L,V),recording(Y,NA1,ACI1,L,V),not empty(L),not empty(V),sim(NA,NA1,S),S>=85.
w(X,Y,19,1) :- eq(X,Y,19).
active(X,Y,19) :- X!=Y,recording(X,NA,ACI,L1,V1),recording(Y,NA1,ACI1,L2,V2),eq(X,RI,_),eq(Y,RI1,_),eq(TI,TI1,0,_),track(TI,RI,MI,P,N,NA2,ACI2,L3,IDT),track(TI1,RI1,MI1,P1,N1,NA3,ACI3,L4,IDT1),sim(NA,NA1,S),S>=85.
w(X,Y,20,1) :- eq(X,Y,20).
active(X,Y,20) :- X!=Y,release(X,RGI,ACI,NA,B,ST,PA,LA,SC,Q),release(Y,RGI1,ACI1,NA1,B1,ST1,PA1,LA1,SC1,Q1),eq(ACI,ACI1,_),sim(NA,NA1,S),S>=90.
w(X,Y,21,1) :- eq(X,Y,21).
active(X,Y,21) :- X!=Y,artist_credit(X,NA,AC,RC,C),artist_credit(Y,NA1,AC,RC,C1),not empty(AC),not empty(RC),eq(X,ACI,_),eq(Y,ACI1,_),artist_credit_name(ACI,P,AI,NA2),artist_credit_name(ACI1,P,AI1,NA3),not empty(P),sim(NA,NA1,S),S>=90,sim(NA2,NA3,S1),S1>=90.
w(X,Y,22,1) :- eq(X,Y,22).
active(X,Y,22) :- X!=Y,label(X,NA,GT,ARI,END1),label(Y,NA1,GT,ARI1,END2),not empty(GT),sim(NA,NA1,S),S>=90.
w(X,Y,23,1) :- eq(X,Y,23).
active(X,Y,23) :- X!=Y,label(X,NA,GT,ARI,END1),label(Y,NA1,GT1,ARI1,END2),sim(NA,NA1,S),S>=95.
w(X,Y,24,1) :- eq(X,Y,24).
active(X,Y,24) :- X!=Y,place(X,NA,PT,ADD1,ARI1,CD,END),place(Y,NA1,PT1,ADD2,ARI2,CD,END1),not empty(CD).
w(X,Y,25,1) :- eq(X,Y,25).
active(X,Y,25) :- X!=Y,place(X,NA,PT,ADD1,ARI1,CD1,END),place(Y,NA1,PT1,ADD2,ARI2,CD2,END1),eq(ARI1,ARI2,_),sim(NA,NA1,S),S>=90.
w(X,Y,26,1) :- eq(X,Y,26).
active(X,Y,26) :- X!=Y,area(X,NA,ART,END),area(Y,NA1,ART,END),not empty(ART),not empty(END),sim(NA,NA1,S),S>=90.
 :- release(X,RGI,AC,NA,B,S,PA,LA,SC,Q),release(Y,RGI1,AC1,NA1,B1,S1,PA1,LA1,SC1,Q1),eq(X,Y,_),B!=B1,not empty(B),not empty(B1).
 :- release(X,RGI,AC,NA,B,S,PA,LA,SC,Q),release(Y,RGI1,AC1,NA1,B1,S1,PA1,LA1,SC1,Q1),eq(X,Y,_),not eq(RGI,RGI1,_).
 :- label(X,NA,GT,ARI,END1),label(Y,NA1,GT1,ARI1,END2),eq(X,Y,_),not eq(ARI,ARI1,_),not empty(ARI),not empty(ARI1).
 :- artist_credit_name(ACI,P,AI,NA2),artist_credit_name(ACI1,P,AI1,NA3),not empty(P),eq(ACI,ACI1,_),not eq(AI,AI1,_).

%!trace_rule{"(%,%) is transitivly closed by (%,%) - (%,%)", X,Z,X,Y,Z,Y}.
eq(X,Y,0) :- eq(X,Z,_),eq(Z,Y,_).
%!trace_rule{"(%,%) is symmetrically closed by (%,%) ", X,Y,X,Y,Y,X}.
eq(X,Y,0) :- eq(Y,X,_).
w(X,Y,0,2) :- eq(X,Y,0).


{eq(X,Y,R)} :- active(X,Y,R).

empty(nan). empty("nan").empty("ーーー").
#show.
#show(track,V0,V9):track(V0,V1,V2,V3,V4,V5,V6,V7,V8),track(V9,V10,V11,V12,V13,V14,V15,V16,V17),eq(V0,V9,RID),V0!=V9.
#show(recording,V0,V5):recording(V0,V1,V2,V3,V4),recording(V5,V6,V7,V8,V9),eq(V0,V5,RID),V0!=V5.
#show(medium,V0,V5):medium(V0,V1,V2,V3,V4),medium(V5,V6,V7,V8,V9),eq(V0,V5,RID),V0!=V5.
#show(artist_credit,V0,V5):artist_credit(V0,V1,V2,V3,V4),artist_credit(V5,V6,V7,V8,V9),eq(V0,V5,RID),V0!=V5.
#show(release_group,V0,V4):release_group(V0,V1,V2,V3),release_group(V4,V5,V6,V7),eq(V0,V4,RID),V0!=V4.
#show(release,V0,V10):release(V0,V1,V2,V3,V4,V5,V6,V7,V8,V9),release(V10,V11,V12,V13,V14,V15,V16,V17,V18,V19),eq(V0,V10,RID),V0!=V10.
#show(artist,V0,V14):artist(V0,V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,V11,V12,V13),artist(V14,V15,V16,V17,V18,V19,V20,V21,V22,V23,V24,V25,V26,V27),eq(V0,V14,RID),V0!=V14.
#show(area,V0,V4):area(V0,V1,V2,V3),area(V4,V5,V6,V7),eq(V0,V4,RID),V0!=V4.
#show(place,V0,V7):place(V0,V1,V2,V3,V4,V5,V6),place(V7,V8,V9,V10,V11,V12,V13),eq(V0,V7,RID),V0!=V7.
#show(label,V0,V5):label(V0,V1,V2,V3,V4),label(V5,V6,V7,V8,V9),eq(V0,V5,RID),V0!=V5.
