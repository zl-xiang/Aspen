eq(X,Y,0):-X!=Y,track(X,RI,MI,P,N,NA,ACI,L,IDT),track(Y,RI1,MI1,P,N1,NA1,ACI1,L1,IDT1),eq(MI,MI1,14),not empty(P),sim(NA,NA1,S),S>=90.
eq(X,Y,0):-X!=Y,track(X,RI,MI,P,N,NA,ACI,L,IDT),track(Y,RI1,MI1,P1,N1,NA1,ACI1,L1,IDT1),eq(ACI,ACI1,19),sim(NA,NA1,S),S>=90.
eq(X,Y,9):-X!=Y,recording(X,NA,ACI,L,V),recording(Y,NA1,ACI1,L,V),eq(ACI,ACI1,19),sim(NA,NA1,S),S>=90,not empty(L),not empty(V).
eq(X,Y,14):-X!=Y,medium(X,REI,P,F,TC),medium(Y,REI1,P,F1,TC1),eq(REI,REI1,28),not empty(P).
eq(X,Y,28):-X!=Y,release(X,RGI,AC,NA,B,S,PA,LA,SC,Q),release(Y,RGI1,AC1,NA1,B,S1,PA1,LA1,SC1,Q1),not empty(B).
eq(X,Y,24):-X!=Y,release_group(X,NA,AC,TY),release_group(Y,NA1,AC1,TY1),eq(X,RGI,24),eq(Y,RGI1,24),eq(REI,REI1,28),release(REI,RGI,AC2,NA2,B2,S2,PA2,LA2,SC2,Q2),release(REI1,RGI1,AC3,NA3,B3,S3,PA3,LA3,SC3,Q3),sim(NA,NA1,S),S>=90.
eq(X,Y,24):-X!=Y,release_group(X,NA,ACI,TY),release_group(Y,NA1,ACI1,TY1),eq(ACI,ACI1,19),eq(X,RGI,24),eq(Y,RGI1,24),eq(REI,REI1,28),release(REI,RGI,AC2,NA2,B2,S2,PA2,LA2,SC2,Q2),release(REI1,RGI1,AC3,NA3,B3,S3,PA3,LA3,SC3,Q3).
eq(X,Y,42):-X!=Y,artist(X,NA,SN,BY,BM,BD,EY,EM,ED,END,AT,G,ARI,C),artist(Y,NA1,SN1,BY1,BM1,BD1,EY1,EM1,ED1,END1,AT1,G1,ARI1,C1),eq(X,AI,42),eq(Y,AI1,42),artist_credit_name(ACI,P,AI,NA4),artist_credit_name(ACI1,P,AI1,NA5),not empty(P),eq(ACI,ACI1,19).
eq(X,Y,42):-X!=Y,artist(X,NA,SN,BY,BM,BD,EY,EM,ED,END,AT,G,ARI,C),artist(Y,NA1,SN1,BY1,BM1,BD1,EY1,EM1,ED1,END1,AT1,G1,ARI1,C1),eq(ARI,ARI1,56),sim(NA,NA1,S),S>=95.
eq(X,Y,56):-X!=Y,area(X,NA,ART,END),area(Y,NA1,ART,END1),not empty(ART),eq(X,ARI,56),eq(Y,ARI1,56),eq(PI,PI1,60),place(PI,NA2,PT,ADD,ARI,CD,END2),place(PI1,NA3,PT1,ADD1,ARI1,CD1,END3),sim(NA,NA1,S),S>=90.
eq(X,Y,67):-X!=Y,label(X,NA,LT,ARI,END1),label(Y,NA1,LT,ARI1,END2),eq(ARI,ARI1,56),not empty(LT),sim(NA,NA1,S),S>=90.
eq(X,Y,60):-X!=Y,place(X,NA,PT,ADD,ARI1,CD,END),place(Y,NA1,PT,ADD1,ARI2,CD1,END1),not empty(PT),eq(ARI1,ARI2,56),sim(NA,NA1,S),S>=90.

active(X,Y,28):-X!=Y,release(X,RGI,AC,NA,B,ST,PA,LA,SC,Q),not empty(ST),release(Y,RGI1,AC1,NA1,B1,ST,PA1,LA1,SC1,Q1),eq(RGI,RGI1,24),sim(NA,NA1,S),S>=90.
active(X,Y,24):-X!=Y,release_group(X,NA2,ACI,TY),release_group(Y,NA3,ACI1,TY1),eq(ACI,ACI1,19),sim(NA2,NA3,S),S>=90.
active(X,Y,42):-X!=Y,artist(X,NA,SN,BY,BM,BD,EY,EM,ED,END,AT,G,ARI,C),not empty(BY),not empty(AT),artist(Y,NA1,SN1,BY,BM1,BD1,EY1,EM1,ED1,END1,AT,G1,ARI1,C1),sim(NA,NA1,S),S>=90.
active(X,Y,42):-X!=Y,artist(X,NA,SN,BY,BM,BD,EY,EM,ED,END,AT,G,ARI,C),not empty(G),artist(Y,NA1,SN1,BY1,BM1,BD1,EY1,EM1,ED1,END1,AT1,G,ARI1,C1),sim(NA,NA1,S),S>=90,sim(SN,SN1,S1),S1>=90.
active(X,Y,0):-X!=Y,track(X,RI,MI,P,N,NA,ACI,L,IDT),track(Y,RI1,MI1,P,N,NA1,ACI1,L,IDT),not empty(P),not empty(N),not empty(L),not empty(IDT),sim(NA,NA1,S),S>=85.
active(X,Y,9):-X!=Y,recording(X,NA,ACI,L,V),recording(Y,NA1,ACI1,L,V),not empty(L),not empty(V),sim(NA,NA1,S),S>=85.
active(X,Y,9):-X!=Y,recording(X,NA,ACI,L1,V1),recording(Y,NA1,ACI1,L2,V2),eq(X,RI,9),eq(Y,RI1,9),eq(TI,TI1,0),track(TI,RI,MI,P,N,NA2,ACI2,L3,IDT),track(TI1,RI1,MI1,P1,N1,NA3,ACI3,L4,IDT1),sim(NA,NA1,S),S>=85.
active(X,Y,28):-X!=Y,release(X,RGI,ACI,NA,B,ST,PA,LA,SC,Q),release(Y,RGI1,ACI1,NA1,B1,ST1,PA1,LA1,SC1,Q1),eq(ACI,ACI1,19),sim(NA,NA1,S),S>=90.
active(X,Y,19):-X!=Y,artist_credit(X,NA,AC,RC,C),artist_credit(Y,NA1,AC,RC,C1),not empty(AC),not empty(RC),eq(X,ACI,19),eq(Y,ACI1,19),artist_credit_name(ACI,P,AI,NA2),artist_credit_name(ACI1,P,AI1,NA3),not empty(P),sim(NA,NA1,S),S>=90,sim(NA2,NA3,S1),S1>=90.
active(X,Y,67):-X!=Y,label(X,NA,GT,ARI,END1),label(Y,NA1,GT,ARI1,END2),not empty(GT),sim(NA,NA1,S),S>=90.
active(X,Y,67):-X!=Y,label(X,NA,GT,ARI,END1),label(Y,NA1,GT1,ARI1,END2),sim(NA,NA1,S),S>=95.
active(X,Y,60):-X!=Y,place(X,NA,PT,ADD1,ARI1,CD,END),place(Y,NA1,PT1,ADD2,ARI2,CD,END1),not empty(CD).
active(X,Y,60):-X!=Y,place(X,NA,PT,ADD1,ARI1,CD1,END),place(Y,NA1,PT1,ADD2,ARI2,CD2,END1),eq(ARI1,ARI2,56),sim(NA,NA1,S),S>=90.
active(X,Y,56):-X!=Y,area(X,NA,ART,END),area(Y,NA1,ART,END),not empty(ART),not empty(END),sim(NA,NA1,S),S>=90.
:-release(X,RGI,AC,NA,B,S,PA,LA,SC,Q),release(Y,RGI1,AC1,NA1,B1,S1,PA1,LA1,SC1,Q1),eq(X,Y,28),B!=B1,not empty(B),not empty(B1).
:-label(X,NA,GT,ARI,END1),label(Y,NA1,GT1,ARI1,END2),eq(X,Y,67),not eq(ARI,ARI1,56),not empty(ARI),not empty(ARI1).

{eq(X,Y,I);neq(X,Y,I)}=1 :- active(X,Y,I).

empty(nan). empty("nan").

eq(X,Y,I) :- eq(X,Z,I),eq(Z,Y,I).
eq(X,Y,I) :- eq(Y,X,I).

#show.
#show(track,V0,V9):track(V0,V1,V2,V3,V4,V5,V6,V7,V8),track(V9,V10,V11,V12,V13,V14,V15,V16,V17),eq(V0,V9,0),V0!=V9.
#show(recording,V0,V5):recording(V0,V1,V2,V3,V4),recording(V5,V6,V7,V8,V9),eq(V0,V5,9),V0!=V5.
#show(medium,V0,V5):medium(V0,V1,V2,V3,V4),medium(V5,V6,V7,V8,V9),eq(V0,V5,14),V0!=V5.
#show(artist_credit,V0,V5):artist_credit(V0,V1,V2,V3,V4),artist_credit(V5,V6,V7,V8,V9),eq(V0,V5,19),V0!=V5.
#show(release_group,V0,V4):release_group(V0,V1,V2,V3),release_group(V4,V5,V6,V7),eq(V0,V4,24),V0!=V4.
#show(release,V0,V10):release(V0,V1,V2,V3,V4,V5,V6,V7,V8,V9),release(V10,V11,V12,V13,V14,V15,V16,V17,V18,V19),eq(V0,V10,28),V0!=V10.
#show(artist,V0,V14):artist(V0,V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,V11,V12,V13),artist(V14,V15,V16,V17,V18,V19,V20,V21,V22,V23,V24,V25,V26,V27),eq(V0,V14,42),V0!=V14.
#show(area,V0,V4):area(V0,V1,V2,V3),area(V4,V5,V6,V7),eq(V0,V4,56),V0!=V4.
#show(place,V0,V7):place(V0,V1,V2,V3,V4,V5,V6),place(V7,V8,V9,V10,V11,V12,V13),eq(V0,V7,60),V0!=V7.
#show(label,V0,V5):label(V0,V1,V2,V3,V4),label(V5,V6,V7,V8,V9),eq(V0,V5,67),V0!=V5.



eq(X,Y,0):-X!=Y,track(X,RI,MI,P,N,NA,ACI,L,IDT),track(Y,RI1,MI1,P,N1,NA1,ACI1,L1,IDT1),eq(MI,MI1,14),not empty(P),sim(NA,NA1,S),S>=90.
eq(X,Y,0):-X!=Y,track(X,RI,MI,P,N,NA,ACI,L,IDT),track(Y,RI1,MI1,P1,N1,NA1,ACI1,L1,IDT1),eq(ACI,ACI1,19),sim(NA,NA1,S),S>=90.
eq(X,Y,9):-X!=Y,recording(X,NA,ACI,L,V),recording(Y,NA1,ACI1,L,V),eq(ACI,ACI1,19),sim(NA,NA1,S),S>=90,not empty(L),not empty(V).
eq(X,Y,14):-X!=Y,medium(X,REI,P,F,TC),medium(Y,REI1,P,F1,TC1),eq(REI,REI1,28),not empty(P).
eq(X,Y,42):-X!=Y,artist(X,NA,SN,BY,BM,BD,EY,EM,ED,END,AT,G,ARI,C),artist(Y,NA1,SN1,BY1,BM1,BD1,EY1,EM1,ED1,END1,AT1,G1,ARI1,C1),eq(ARI,ARI1,56),sim(NA,NA1,S),S>=95.
eq(X,Y,56):-X!=Y,area(X,NA,ART,END),area(Y,NA1,ART,END1),not empty(ART),eq(X,ARI,56),eq(Y,ARI1,56),eq(PI,PI1,60),place(PI,NA2,PT,ADD,ARI,CD,END2),place(PI1,NA3,PT1,ADD1,ARI1,CD1,END3),sim(NA,NA1,S),S>=90.
eq(X,Y,67):-X!=Y,label(X,NA,LT,ARI,END1),label(Y,NA1,LT,ARI1,END2),eq(ARI,ARI1,56),not empty(LT),sim(NA,NA1,S),S>=90.
eq(X,Y,60):-X!=Y,place(X,NA,PT,ADD,ARI1,CD,END),place(Y,NA1,PT,ADD1,ARI2,CD1,END1),not empty(PT),eq(ARI1,ARI2,56),sim(NA,NA1,S),S>=90.
%!trace_rule{" release (%,%)-(%,%) merged since they have the same barcode % ", X,REI,Y,REI1,B}.
eq(X,Y,28):-X!=Y,release(X,RGI,AC,NA,B,S,PA,LA,SC,Q),release(Y,RGI1,AC1,NA1,B,S1,PA1,LA1,SC1,Q1),not empty(B).
%!trace_rule{" release_group (%,%)-(%,%) are merged since they have similar names (%,%):%  and contain the same release (%,%)-(%,%) ", X,RGI,Y,RGI1,NA,NA1,S,T,REI,T1,REI1}.
eq(X,Y,24):-X!=Y,release_group(X,NA,AC,TY),release_group(Y,NA1,AC1,TY1),eq(X,RGI,24),eq(Y,RGI1,24),eq(REI,REI1,28),release(REI,RGI,AC2,NA2,B2,S2,PA2,LA2,SC2,Q2),release(REI1,RGI1,AC3,NA3,B3,S3,PA3,LA3,SC3,Q3),sim(NA,NA1,S),S>=90.
%!trace_rule{" release_group (%,%)-(%,%) are merged since they contain the same release (%,%)-(%,%) and have the same artist credits (%,%)-(%,%)", X,RGI,Y,RGI1,T,REI,T1,REI1,T3,ACI,T4,ACI1}.
eq(X,Y,24):-X!=Y,release_group(X,NA,ACI,TY),release_group(Y,NA1,ACI1,TY1),eq(ACI,ACI1,19),eq(X,RGI,24),eq(Y,RGI1,24),eq(REI,REI1,28),release(REI,RGI,AC2,NA2,B2,S2,PA2,LA2,SC2,Q2),release(REI1,RGI1,AC3,NA3,B3,S3,PA3,LA3,SC3,Q3).
%!trace_rule{" artist (%,%)-(%,%) are merged since they have join on (artist_credit_name.position:%) and belong to the same artist_credit (%,%)-(%,%)", X,AI,Y,AI1,P,T3,ACI,T4,ACI1}.
eq(X,Y,42):-X!=Y,artist(X,NA,SN,BY,BM,BD,EY,EM,ED,END,AT,G,ARI,C),artist(Y,NA1,SN1,BY1,BM1,BD1,EY1,EM1,ED1,END1,AT1,G1,ARI1,C1),eq(X,AI,42),eq(Y,AI1,42),artist_credit_name(ACI,P,AI,NA4),artist_credit_name(ACI1,P,AI1,NA5),not empty(P),eq(ACI,ACI1,19).
active(X,Y,42):-X!=Y,artist(X,NA,SN,BY,BM,BD,EY,EM,ED,END,AT,G,ARI,C),not empty(BY),not empty(AT),artist(Y,NA1,SN1,BY,BM1,BD1,EY1,EM1,ED1,END1,AT,G1,ARI1,C1),sim(NA,NA1,S),S>=90.
active(X,Y,42):-X!=Y,artist(X,NA,SN,BY,BM,BD,EY,EM,ED,END,AT,G,ARI,C),not empty(G),artist(Y,NA1,SN1,BY1,BM1,BD1,EY1,EM1,ED1,END1,AT1,G,ARI1,C1),sim(NA,NA1,S),S>=90,sim(SN,SN1,S1),S1>=90.
active(X,Y,0):-X!=Y,track(X,RI,MI,P,N,NA,ACI,L,IDT),track(Y,RI1,MI1,P,N,NA1,ACI1,L,IDT),not empty(P),not empty(N),not empty(L),not empty(IDT),sim(NA,NA1,S),S>=85.
active(X,Y,9):-X!=Y,recording(X,NA,ACI,L,V),recording(Y,NA1,ACI1,L,V),not empty(L),not empty(V),sim(NA,NA1,S),S>=85.
active(X,Y,9):-X!=Y,recording(X,NA,ACI,L1,V1),recording(Y,NA1,ACI1,L2,V2),eq(X,RI,9),eq(Y,RI1,9),eq(TI,TI1,0),track(TI,RI,MI,P,N,NA2,ACI2,L3,IDT),track(TI1,RI1,MI1,P1,N1,NA3,ACI3,L4,IDT1),sim(NA,NA1,S),S>=85.
active(X,Y,67):-X!=Y,label(X,NA,GT,ARI,END1),label(Y,NA1,GT,ARI1,END2),not empty(GT),sim(NA,NA1,S),S>=90.
active(X,Y,67):-X!=Y,label(X,NA,GT,ARI,END1),label(Y,NA1,GT1,ARI1,END2),sim(NA,NA1,S),S>=95.
active(X,Y,60):-X!=Y,place(X,NA,PT,ADD1,ARI1,CD,END),place(Y,NA1,PT1,ADD2,ARI2,CD,END1),not empty(CD).
active(X,Y,60):-X!=Y,place(X,NA,PT,ADD1,ARI1,CD1,END),place(Y,NA1,PT1,ADD2,ARI2,CD2,END1),eq(ARI1,ARI2,56),sim(NA,NA1,S),S>=90.
%!trace_rule{" release (%,%)-(%,%) are active since they have similar names (%,%):% and join on (status:%) and belong to the same release_group (%,%)-(%,%) ", X,REI,Y,REI1,NA,NA1,S,ST,T,RGI,T1,RGI1}.
active(X,Y,28):-X!=Y,release(X,RGI,AC,NA,B,ST,PA,LA,SC,Q),not empty(ST),release(Y,RGI1,AC1,NA1,B1,ST,PA1,LA1,SC1,Q1),eq(RGI,RGI1,24),sim(NA,NA1,S),S>=90.
%!trace_rule{" release_group (%,%)-(%,%) actived because of similar names (%,%):% and have the same list of artist credits (%,%)-(%,%) ", X,RGI,Y,RGI1,NA2,NA3,S,T,ACI,T1,ACI1}.
active(X,Y,24):-X!=Y,release_group(X,NA2,ACI,TY),release_group(Y,NA3,ACI1,TY1),eq(ACI,ACI1,19),sim(NA2,NA3,S),S>=90.
%!trace_rule{" release (%,%)-(%,%) are active since they have similar names (%,%):% have the same artist_credit (%,%)-(%,%) ", X,REI,Y,REI1,NA,NA1,S,T,ACI,T1,ACI1}.
active(X,Y,28):-X!=Y,release(X,RGI,ACI,NA,B,ST,PA,LA,SC,Q),release(Y,RGI1,ACI1,NA1,B1,ST1,PA1,LA1,SC1,Q1),eq(ACI,ACI1,19),sim(NA,NA1,S),S>=90.
%!trace_rule{" artist_credit (%,%)-(%,%) actived since they have similar names (%,%):% and join on (artist_count %,reference_count:%) and have similar artist credit names (%,%):% on position (%)", X,AI,Y,AI1,NA,NA1,S,AC,RC,NA2,NA3,S1,P}.
active(X,Y,19):-X!=Y,artist_credit(X,NA,AC,RC,C),artist_credit(Y,NA1,AC,RC,C1),not empty(AC),not empty(RC),eq(X,ACI,19),eq(Y,ACI1,19),artist_credit_name(ACI,P,AI,NA2),artist_credit_name(ACI1,P,AI1,NA3),not empty(P),sim(NA,NA1,S),S>=90,sim(NA2,NA3,S1),S1>=90.
%!trace_rule{" area (%,%)-(%,%) actived since they have similar names (%,%):% and join on (area type:%, end:%)", X,ARI1,Y,ARI2,NA,NA1,S,ART,END}.
active(X,Y,56):-X!=Y,area(X,NA,ART,END),area(Y,NA1,ART,END),not empty(ART),not empty(END),sim(NA,NA1,S),S>=90.