#program base.
#show.
#show(track,TI,TI1):track(X,TI,GID,RI,MI,P,N,NA,ACI,L,IDT),track(Y,TI1,GID1,RI1,MI1,P1,N1,NA1,ACI1,L1,IDT1),eq(X,Y),TI!=TI1.
#show(artist_credit,ACI,ACI1):artist_credit(X,ACI,NA2,ACN,RC,C1,GID2),artist_credit(Y,ACI1,NA3,ACN1,RC1,C2,GID3),eq(X,Y),ACI!=ACI1.
#show(medium,MI,MI1):medium(X,MI,REI,P2,F,TC),medium(Y,MI1,REI1,P3,F2,TC1),eq(X,Y),MI!=MI1.
#show(recording,RI,RI1):recording(X,RI,GID,NA,ACI,L,V),recording(Y,RI1,GID1,NA1,ACI1,L1,V1),eq(X,Y),RI!=RI1.
#show(release,REI,REI1):release(X,REI,GID1,RGI1,AC1,NA1,B1,S1,PA1,LA1,SC1,Q1),release(Y,REI1,GID,RGI,AC,NA,B,S,PA,LA,SC,Q),eq(X,Y),REI1!=REI.
#show(artist,AI,AI1):artist(X,AI,GID,NA,SN,BY,BM,BD,EY,EM,ED,END,AT,G,ARI,C),artist(Y,AI1,GID1,NA1,SN1,BY1,BM1,BD1,EY1,EM1,ED1,END1,AT1,G1,ARI1,C1),eq(X,Y),AI!=AI1.
#show(area,ARI,ARI1):area(X,ARI,GID,NA,ART,END),area(Y,ARI1,GID1,NA1,ART1,END1),eq(X,Y),ARI!=ARI1.
#show(release_group,RGI,RGI1):release_group(X,RGI,GID2,NA2,AC2,TY1),release_group(Y,RGI1,GID3,NA3,AC3,TY2),eq(X,Y),RGI1!=RGI.
#show(place,PI,PI1):place(X,PI,GID,NA,PT,ADD,ARI,CD,END),place(Y,PI1,GID2,NA2,PT1,ADD1,ARI1,CD1,END1),eq(X,Y),PI!=PI1.
#show(label,LI,LI1):label(X,LI,GID,NA,GT,ARI,END1),label(Y,LI1,GID1,NA1,GT1,ARI1,END2),eq(X,Y),LI!=LI1.
empty("nan").
empty(nan).

% eq(X,X):-adom(X).

%!trace_rule{"(%,%) is symmetrically closed by (%,%)", X,Y,Y,X}.
eq(X,Y):-eq(Y,X),X!=Y.

%!trace_rule{"(%,%) is transitivly closed by (%,%) - (%,%) ", X,Z,X,Y,Y,Z}.
eq(X,Z):-eq(X,Y),eq(Y,Z),X!=Z.


{eq(X,Y);neq(X,Y)}=1 :- active(X,Y).

:- eq(X,Y), neq(X,Y).
:- eq(Y,X), neq(X,Y).
% h1
eq(X,Y):-X!=Y,track(X,TI,GID,RI,MI,P,N,NA,ACI,L,IDT),track(Y,TI1,GID1,RI1,MI1,P,N1,NA1,ACI1,L1,IDT1),medium(T,MI,REI,P2,F,TC1),medium(T1,MI1,REI1,P3,F2,TC2),eq(T,T1),not empty(P),sim(NA,NA1,S), S>=90.
% h2
eq(X,Y):-X!=Y,track(X,TI,GID,RI,MI,P,N,NA,ACI,L,IDT),track(Y,TI1,GID1,RI1,MI1,P1,N1,NA1,ACI1,L1,IDT1),artist_credit(T,ACI,NA2,ACN1,RC1,C1,GID2),artist_credit(T1,ACI1,NA3,ACN2,RC2,C2,GID3),eq(T,T1), sim(NA,NA1,S), S>=90.
% h3
eq(X,Y):-X!=Y,recording(X,RI,GID,NA,ACI,L,V),recording(Y,RI1,GID1,NA1,ACI1,L,V),artist_credit(T,ACI,NA2,ACN1,RC1,C1,GID2),artist_credit(T1,ACI1,NA3,ACN2,RC2,C2,GID3),eq(T,T1),
        sim(NA,NA1,S), S>=90, not empty(L), not empty(V).

%trace_rule{"medium (%,%)-(%,%) are merged since they are of the same position (%) of the same release (%,%)-(%,%) ", X,MI,Y,MI1,P,T,REI,T1,REI1}.
eq(X,Y):-X!=Y,medium(X,MI,REI,P,F,TC),medium(Y,MI1,REI1,P,F1,TC1),release(T,REI,GID1,RGI1,AC1,NA1,B1,S1,PA1,LA1,SC1,Q1),release(T1,REI1,GID,RGI,AC,NA,B,S,PA,LA,SC,Q),eq(T,T1),not empty(P).

% case h1
%!trace_rule{" release (%,%)-(%,%) merged since they have the same barcode % ", X,REI,Y,REI1,B}.
eq(X,Y):-X!=Y,release(X,REI,GID,RGI,AC,NA,B,S,PA,LA,SC,Q),release(Y,REI1,GID1,RGI1,AC1,NA1,B,S1,PA1,LA1,SC1,Q1),not empty(B).

% case h2
%!trace_rule{" release_group (%,%)-(%,%) are merged since they have similar names (%,%):%  and contain the same release (%,%)-(%,%) ", X,RGI,Y,RGI1,NA,NA1,S,T,REI,T1,REI1}.
eq(X,Y):-X!=Y,release_group(X,RGI,GID,NA,AC,TY),release_group(Y,RGI1,GID1,NA1,AC1,TY1),
    eq(T,T1),release(T,REI,GID2,RGI,AC2,NA2,B2,S2,PA2,LA2,SC2,Q2),release(T1,REI1,GID3,RGI1,AC3,NA3,B3,S3,PA3,LA3,SC3,Q3), sim(NA,NA1,S), S>=90.


%!trace_rule{" release_group (%,%)-(%,%) are merged since they contain the same release (%,%)-(%,%) and have the same artist credits (%,%)-(%,%)", X,RGI,Y,RGI1,T,REI,T1,REI1,T3,ACI,T4,ACI1}.
eq(X,Y):-X!=Y,release_group(X,RGI,GID,NA,ACI,TY),release_group(Y,RGI1,GID1,NA1,ACI1,TY1),  artist_credit(T3,ACI,NA6,ACN,RC,C4,GID4),artist_credit(T4,ACI1,NA7,ACN1,RC1,C5,GID5),eq(T3,T4),
    eq(T,T1),release(T,REI,GID2,RGI,AC2,NA2,B2,S2,PA2,LA2,SC2,Q2),release(T1,REI1,GID3,RGI1,AC3,NA3,B3,S3,PA3,LA3,SC3,Q3).

%!trace_rule{" artist (%,%)-(%,%) are merged since they have join on (artist_credit_name.position:%) and belong to the same artist_credit (%,%)-(%,%)", X,AI,Y,AI1,P,T3,ACI,T4,ACI1}.
eq(X,Y):-X!=Y,artist(X,AI,GID,NA,SN,BY,BM,BD,EY,EM,ED,END,AT,G,ARI,C),artist(Y,AI1,GID1,NA1,SN1,BY1,BM1,BD1,EY1,EM1,ED1,END1,AT1,G1,ARI1,C1),
        artist_credit_name(T,ACI,P,AI,NA4),artist_credit_name(T2,ACI1,P,AI1,NA5), not empty(P),
        artist_credit(T3,ACI,NA6,ACN,RC,C4,GID2),artist_credit(T4,ACI1,NA7,ACN1,RC1,C5,GID3),eq(T3,T4).


%trace_rule{" artist (%,%)-(%,%) are merged since they have similar names (%,%):% and belong to the same area (%,%)-(%,%) ", X,AI,Y,AI1,NA,NA1,S,T1,ARI,T2,ARI1}.
eq(X,Y):-X!=Y,artist(X,AI,GID,NA,SN,BY,BM,BD,EY,EM,ED,END,AT,G,ARI,C),
    artist(Y,AI1,GID1,NA1,SN1,BY1,BM1,BD1,EY1,EM1,ED1,END1,AT1,G1,ARI1,C1),area(T1,ARI,GID2,NA2,ART1,END2),area(T2,ARI1,GID3,NA3,ART2,END3),eq(T1,T2), sim(NA,NA1,S), S>=95.

%trace_rule{" area (%,%)-(%,%) are merged since they have similar names (%,%):% and join on (area type:%) and contain the same place (%,%)-(%,%) ", X,ARI,Y,ARI1,NA,NA1,S,ART,T,PI,T1,PI1}.
eq(X,Y):-X!=Y,area(X,ARI,GID,NA,ART,END),area(Y,ARI1,GID1,NA1,ART,END1),not empty(ART),
    place(T,PI,GID2,NA2,PT,ADD,ARI,CD,END2),place(T1,PI1,GID3,NA3,PT1,ADD1,ARI1,CD1,END3),eq(T,T1), sim(NA,NA1,S), S>=90.

eq(X,Y):-X!=Y,label(X,LI,GID,NA,LT,ARI,END1),label(Y,LI1,GID1,NA1,LT,ARI1,END2),eq(A1,A2),area(A1,ARI,GID2,NA2,ART,END3),area(A2,ARI1,GID3,NA3,ART1,END4), not empty(LT), sim(NA,NA1,S), S>=90.

eq(X,Y):-X!=Y,place(X,PI,GID,NA,PT,ADD,ARI1,CD,END),place(Y,PI1,GID1,NA1,PT,ADD1,ARI2,CD1,END1), not empty(PT),
    area(A1,ARI1,GID2,NA2,ART,END3),area(A2,ARI2,GID3,NA3,ART1,END4),eq(A1,A2), sim(NA,NA1,S), S>=90.


% [MOD] change to soft rule
% case s1
%!trace_rule{" release (%,%)-(%,%) are active since they have similar names (%,%):% and join on (status:%) and belong to the same release_group (%,%)-(%,%) ", X,REI,Y,REI1,NA,NA1,S,ST,T,RGI,T1,RGI1}.
active(X,Y):-X!=Y,release(X,REI,GID,RGI,AC,NA,B,ST,PA,LA,SC,Q), not empty(ST),
release(Y,REI1,GID1,RGI1,AC1,NA1,B1,ST,PA1,LA1,SC1,Q1),
release_group(T,RGI,GID2,NA2,AC2,TY1),release_group(T1,RGI1,GID3,NA3,AC3,TY2),eq(T,T1), sim(NA,NA1,S), S>=90.

%!trace_rule{" release_group (%,%)-(%,%) actived because of similar names (%,%):% and have the same list of artist credits (%,%)-(%,%) ", X,RGI,Y,RGI1,NA2,NA3,S,T,ACI,T1,ACI1}.
active(X,Y):-X!=Y,release_group(X,RGI,GID2,NA2,ACI,TY),release_group(Y,RGI1,GID3,NA3,ACI1,TY1), 
    artist_credit(T,ACI,NA,AC,RC,C,GID),artist_credit(T1,ACI1,NA1,AC1,RC1,C1,GID1), eq(T,T1), sim(NA2,NA3,S), S>=90.

%trace_rule{" artist (%,%)-(%,%) actived since they have similar names (%,%):% and join on (begin date %,artist_type:%)", X,AI,Y,AI1,NA,NA1,S,BY,AT}.
active(X,Y):-X!=Y,artist(X,AI,GID,NA,SN,BY,BM,BD,EY,EM,ED,END,AT,G,ARI,C),not empty(BY),not empty(AT),
        artist(Y,AI1,GID1,NA1,SN1,BY,BM1,BD1,EY1,EM1,ED1,END1,AT,G1,ARI1,C1), sim(NA,NA1,S), S>=90.

%trace_rule{" artist (%,%)-(%,%) actived since they have similar names (%,%):% and similar sort name (%,%):% and join on (begin date %,artist_type:%)", X,AI,Y,AI1,NA,NA1,S,SN,SN1,S1,BY,AT}.
active(X,Y):-X!=Y,artist(X,AI,GID,NA,SN,BY,BM,BD,EY,EM,ED,END,AT,G,ARI,C), not empty(G),
artist(Y,AI1,GID1,NA1,SN1,BY1,BM1,BD1,EY1,EM1,ED1,END1,AT1,G,ARI1,C1), sim(NA,NA1,S), S>=90, sim(SN,SN1,S1), S1>=90.

active(X,Y):-X!=Y,track(X,TI,GID,RI,MI,P,N,NA,ACI,L,IDT),track(Y,TI1,GID1,RI1,MI1,P,N,NA1,ACI1,L,IDT),
    not empty(P),not empty(N),not empty(L),not empty(IDT), sim(NA,NA1,S), S>=85.

active(X,Y):-X!=Y,recording(X,RI,GID,NA,ACI,L,V),recording(Y,RI1,GID1,NA1,ACI1,L,V),
    not empty(L),not empty(V), sim(NA,NA1,S), S>=85.

active(X,Y):-X!=Y,recording(X,RI,GID,NA,ACI,L1,V1),recording(Y,RI1,GID1,NA1,ACI1,L2,V2),
    track(T,TI,GID2,RI,MI,P,N,NA2,ACI2,L3,IDT),track(T1,TI1,GID3,RI1,MI1,P1,N1,NA3,ACI3,L4,IDT1), eq(T,T1), sim(NA,NA1,S), S>=80.


% case s3
%!trace_rule{" release (%,%)-(%,%) are active since they have similar names (%,%):% have the same artist_credit (%,%)-(%,%) ", X,REI,Y,REI1,NA,NA1,S,T,ACI,T1,ACI1}.
active(X,Y):-X!=Y,release(X,REI,GID,RGI,ACI,NA,B,ST,PA,LA,SC,Q),
release(Y,REI1,GID1,RGI1,ACI1,NA1,B1,ST1,PA1,LA1,SC1,Q1),
artist_credit(T,ACI,NA2,AC,RC,C,GID2),artist_credit(T1,ACI1,NA3,AC1,RC1,C1,GID3),eq(T,T1), sim(NA,NA1,S), S>=90.

% case s2
%!trace_rule{" artist_credit (%,%)-(%,%) actived since they have similar names (%,%):% and join on (artist_count %,reference_count:%) and have similar artist credit names (%,%):% on position (%)", X,AI,Y,AI1,NA,NA1,S,AC,RC,NA2,NA3,S1,P}.
active(X,Y):-X!=Y,artist_credit(X,ACI,NA,AC,RC,C,GID),artist_credit(Y,ACI1,NA1,AC,RC,C1,GID1),
    not empty(AC),not empty(RC),artist_credit_name(T,ACI,P,AI,NA2),artist_credit_name(T2,ACI1,P,AI1,NA3),not empty(P), sim(NA,NA1,S), S>=90, sim(NA2,NA3,S1), S1>=90.
%---
active(X,Y):-X!=Y,label(X,LI,GID,NA,GT,ARI,END1),label(Y,LI1,GID1,NA1,GT,ARI1,END2),not empty(GT), sim(NA,NA1,S), S>=90.

active(X,Y):-X!=Y,label(X,LI,GID,NA,GT,ARI,END1),label(Y,LI1,GID1,NA1,GT1,ARI1,END2), sim(NA,NA1,S), S>=95.

active(X,Y):-X!=Y,place(X,PI,GID,NA,PT,ADD1,ARI1,CD,END),place(Y,PI1,GID1,NA1,PT1,ADD2,ARI2,CD,END1), not empty(CD).

active(X,Y):-X!=Y,place(X,PI,GID,NA,PT,ADD1,ARI1,CD1,END),place(Y,PI1,GID1,NA1,PT1,ADD2,ARI2,CD2,END1),
        eq(A1,A2),area(A1,ARI1,GID2,NA2,ART,END3),area(A2,ARI2,GID3,NA3,ART1,END4), sim(NA,NA1,S), S>=90.

%!trace_rule{" area (%,%)-(%,%) actived since they have similar names (%,%):% and join on (area type:%, end:%)", X,ARI1,Y,ARI2,NA,NA1,S,ART,END}.
active(X,Y):-X!=Y,area(X,ARI1,GID2,NA,ART,END),area(Y,ARI2,GID3,NA1,ART,END),not empty(ART),not empty(END), sim(NA,NA1,S), S>=90.

% reject the release merge where the releases have different barcodes if not empty
%!trace_rule{" release merge (%,%)-(%,%) is rejected since they have different barcodes (%,%)", X,REI,Y,REI1,B,B1}.
falsify(X,Y):- release(X,REI,GID,RGI,AC,NA,B,S,PA,LA,SC,Q), release(Y,REI1,GID1,RGI1,AC1,NA1,B1,S1,PA1,LA1,SC1,Q1), eq(X,Y), B!=B1, not empty(B), not empty(B1).

%!trace_rule{" release merge (%,%)-(%,%) is rejected since they have different release group (%,%)-(%,%)", X,REI,Y,REI1,B,B1,RG,RGI,RG1,RGI1}.
falsify(X,Y):- release(X,REI,GID,RGI,AC,NA,B,S,PA,LA,SC,Q), release(Y,REI1,GID1,RGI1,AC1,NA1,B1,S1,PA1,LA1,SC1,Q1),  eq(X,Y), 
    release_group(RG,RGI,GID2,NA2,AC2,TY),release_group(RG1,RGI1,GID3,NA3,AC3,TY1),  not eq(RG,RG1).

%trace_rule{" label merge (%,%)-(%,%) is rejected since they belong to different area group (%,%)-(%,%)", X,LI,Y,LI1,A1,ARI,A2,ARI1}.
falsify(X,Y):- label(X,LI,GID,NA,GT,ARI,END1),label(Y,LI1,GID1,NA1,GT1,ARI1,END2), eq(X,Y), not empty(ARI), not empty(ARI1), 
           not eq(A1,A2),area(A1,ARI,GID2,NA2,ART,END3),area(A2,ARI1,GID3,NA3,ART1,END4).

:- falsify(X,Y),eq(X,Y).