
%%%%%%%%%%% hard rules
% it is a merge if two track-tuples have the same artist_credit eq(ACI,ACI1) and have similar names NA, NA1
eq(X,Y):-X!=Y,track(X,RI,MI,P,N,NA,ACI,L,IDT),track(Y,RI1,MI1,P1,N1,NA1,ACI1,L1,IDT1),eq(ACI,ACI1),sim(NA,NA1,S),S>=95.

% track(id, recording, medium, position, number, name, artist_credit, length, is_data_track)
% it is a merge if two track-tuples on the same position P of the same medium eq(MI,MI1), and they have similar names NA, NA1
eq(X,Y):-X!=Y,track(X,RI,MI,P,N,NA,ACI,L,IDT),track(Y,RI1,MI1,P,N1,NA1,ACI1,L1,IDT1),eq(MI,MI1),not empty(P),sim(NA,NA1,S),S>=95.


% it is a merge if two track-tuples on the same position P of the same medium eq(MI,MI1), and they have similar names NA, NA1
% eq(X,Y):-X!=Y,track(X,RI,MI,P,N,NA,ACI,L,IDT),track(Y,RI1,MI1,P,N1,NA1,ACI1,L1,IDT1),eq(RI,RI1),not empty(P),sim(NA,NA1,S),S>=95.

% recording(id, name, artist_credit, length, video)
% it is a merge if two recording-tuples have the same artist_credit eq(ACI,ACI1) and similar names and the same length L and is_video V
eq(X,Y):-X!=Y,recording(X,NA,ACI,L,V),recording(Y,NA1,ACI1,L,V),eq(ACI,ACI1),sim(NA,NA1,S),S>=90,not empty(L),not empty(V).




% medium(id, release, position, format, track_count,)
% it is a merge if two medium-tuples have the same release eq(REI,REI1) and are of the same position P
eq(X,Y):-X!=Y,medium(X,REI,P,F,TC),medium(Y,REI1,P,F1,TC1),eq(REI,REI1),not empty(P).

% release(id, release_group, artist_credit, name, barcode, status, packaging, language, script, quality,)
% it is a merge if two release-tuples have the same barcode
eq(X,Y):-X!=Y,release(X,RGI,AC,NA,B,S,PA,LA,SC,Q),release(Y,RGI1,AC1,NA1,B,S1,PA1,LA1,SC1,Q1),not empty(B).

% release_group(id, name, artist_credit, type,)
% it is a merge if two release_group-tuples have the same release eq(REI,REI1)
eq(X,Y):-X!=Y,release_group(X,NA,AC,TY),release_group(Y,NA1,AC1,TY1),eq(X,RGI),eq(Y,RGI1),eq(REI,REI1),release(REI,RGI,AC2,NA2,B2,S2,PA2,LA2,SC2,Q2),release(REI1,RGI1,AC3,NA3,B3,S3,PA3,LA3,SC3,Q3).

% it is a merge if two release_group-tuples have the same release eq(REI,REI1) and the same artist_credit eq(ACI,ACI1)
eq(X,Y):-X!=Y,release_group(X,NA,ACI,TY),release_group(Y,NA1,ACI1,TY1),eq(ACI,ACI1),eq(X,RGI),eq(Y,RGI1),eq(REI,REI1),release(REI,RGI,AC2,NA2,B2,S2,PA2,LA2,SC2,Q2),release(REI1,RGI1,AC3,NA3,B3,S3,PA3,LA3,SC3,Q3).

% artist(id, name, sort_name, begin_date_year, begin_date_month, begin_date_day, end_date_year, end_date_month, end_date_day, ended, artist_type, gender, area, comment, )
% it is a merge if two artist-tuples appear in the same artist_credit eq(ACI,ACI1) and are of the same position P
eq(X,Y):-X!=Y,artist(X,NA,SN,BY,BM,BD,EY,EM,ED,END,AT,G,ARI,C),artist(Y,NA1,SN1,BY1,BM1,BD1,EY1,EM1,ED1,END1,AT1,G1,ARI1,C1),eq(X,AI),eq(Y,AI1),artist_credit_name(ACI,P,AI,NA4),artist_credit_name(ACI1,P,AI1,NA5),not empty(P),eq(ACI,ACI1).

% area(id, name, area_type, ended)
% it is a merge if two area-tuples have the same area type ART, and have the same place eq(PI,PI1), and have similar names NA,NA1
eq(X,Y):-X!=Y,area(X,NA,ART,END),area(Y,NA1,ART1,END1),eq(X,ARI),eq(Y,ARI1),eq(PI,PI1),place(PI,NA2,PT,ADD,ARI,CD,END2),place(PI1,NA3,PT1,ADD1,ARI1,CD1,END3).


% area(id, name, area_type, ended)
% it is a merge if two area-tuples have the same area type ART, and have the same place eq(PI,PI1), and have similar names NA,NA1
eq(X,Y):-X!=Y,area(X,NA,ART,END),area(Y,NA1,ART1,END1),eq(X,ARI),eq(Y,ARI1),eq(LI,LI1),label(LI1,NA2,LT,ARI,END1),label(LI1,NA3,LT1,ARI1,END2).


% label(id, name, label_type, area, ended,)
% it is a merge if two label-tuples belong to the same area eq(ARI,ARI1) and have the same label type LT, and have similar names NA,NA1
eq(X,Y):-X!=Y,label(X,NA,LT,ARI,END1),label(Y,NA1,LT,ARI1,END2),eq(ARI,ARI1),not empty(LT),sim(NA,NA1,S),S>=90.

% place(id, name, place_type, address, area, coordinates, ended,)
% it is a merge if two place-tuples belong to the same area eq(ARI1,ARI2) and have the same place type PT, and have similar names NA,NA1
eq(X,Y):-X!=Y,place(X,NA,PT,ADD,ARI1,CD,END),place(Y,NA1,PT,ADD1,ARI2,CD1,END1),not empty(PT),eq(ARI1,ARI2),sim(NA,NA1,S),S>=90.

%eq(X,Y):-X!=Y,place(X,NA,PT,ADD,ARI1,CD,END),place(Y,NA1,PT,ADD1,ARI2,CD1,END1),not empty(PT),eq(ARI1,ARI2),sim(NA,NA1,S),S>=90.
%%%%%%%%%%%%

% [ADDED] when similarity join is invalidated for track
eq(X,Y):-X!=Y,track(X,RI,MI,P,N,NA,ACI,L,IDT),track(Y,RI1,MI1,P,N1,NA1,ACI1,L1,IDT1),
    eq(MI,MI1),not empty(P), eq(ACI,ACI1). 

% [ADDED]  invalidated similarity, add rule check artist and release
eq(X,Y):-X!=Y,artist_credit(X,NA,AC,RC,C),artist_credit(Y,NA1,AC1,RC1,C1),
recording(RE,NA2,X1,L,V),recording(RE1,NA3,Y1,L1,V1), eq(RE,RE1), eq(X,X1), eq(Y,Y1).

% [ADDED]  invalidated similarity, add rule check artist and release
eq(X,Y):-X!=Y,artist_credit(X,NA,AC,RC,C),artist_credit(Y,NA1,AC1,RC1,C1),
release_group(RG,NA2,X1,TY),release_group(RG1,NA3,Y1,TY1),eq(RG,RG1), eq(X,X1), eq(Y,Y1).

% [ADDED]  invalidated similarity, add rule check artist and release
eq(X,Y):-X!=Y,artist_credit(X,NA,AC,RC,C),artist_credit(Y,NA1,AC1,RC1,C1),
release(REI,RGI,X1,NA5,B,ST,PA,LA,SC,Q), 
release(REI1,RGI1,Y1,NA6,B1,ST1,PA1,LA1,SC1,Q1), eq(REI,REI1), eq(X,X1), eq(Y,Y1).

% [ADDED]  invalidated similarity, add rule check artist and release
eq(X,Y):-X!=Y,artist_credit(X,NA,AC,RC,C),artist_credit(Y,NA1,AC1,RC1,C1),
track(TR,RI,MI,P,N,NA2,X1,L,IDT),track(TR1,RI1,MI1,P1,N1,NA3,Y1,L1,IDT1), eq(TR,TR1), eq(X,X1), eq(Y,Y1).

%%%%%%%%%%% soft rules
% it is likely a merge if two artist-tuples are in the same area eq(ARI,ARI1) and have simialr names NA,NA1
active(X,Y):-X!=Y,artist(X,NA,SN,BY,BM,BD,EY,EM,ED,END,AT,G,ARI,C),artist(Y,NA1,SN1,BY1,BM1,BD1,EY1,EM1,ED1,END1,AT1,G1,ARI1,C1),eq(ARI,ARI1),sim(NA,NA1,S),S>=95.

% it is likely a merge if two release-tuples belong to the same release_group eq(RGI,RGI1) and have similar names NA,NA1
active(X,Y):-X!=Y,release(X,RGI,AC,NA,B,ST,PA,LA,SC,Q),not empty(ST),release(Y,RGI1,AC1,NA1,B1,ST,PA1,LA1,SC1,Q1),eq(RGI,RGI1),sim(NA,NA1,S),S>=90.

% it is likely a merge if two artist-tuples have the same born year BY and the same artist type AT and have similar names NA,NA1,
active(X,Y):-X!=Y,artist(X,NA,SN,BY,BM,BD,EY,EM,ED,END,AT,G,ARI,C),not empty(BY),not empty(AT),artist(Y,NA1,SN1,BY,BM1,BD1,EY1,EM1,ED1,END1,AT,G1,ARI1,C1),sim(NA,NA1,S),S>=90.

% it is likely a merge if two artist-tuples have the same gender G and have similar names NA,NA1 and similar sort_name SN,SN1
% active(X,Y):-X!=Y,artist(X,NA,SN,BY,BM,BD,EY,EM,ED,END,AT,G,ARI,C),not empty(G),artist(Y,NA1,SN1,BY1,BM1,BD1,EY1,EM1,ED1,END1,AT1,G,ARI1,C1),sim(NA,NA1,S),S>=95,sim(SN,SN1,S1),S1>=95.


%active(X,Y):-X!=Y,artist(X,NA,SN,BY,BM,BD,EY,EM,ED,END,AT,G,ARI,C), not empty(G), 
%artist(Y,NA1,SN1,BY1,BM1,BD1,EY1,EM1,ED1,END1,AT1,G,ARI1,C1), sim(NA,NA1,S), S>=90, eq(ARI,ARI1).



%---
% [ADDED] make soft,  add rule check track and reference count
active(X,Y):-X!=Y,artist_credit(X,NA,AC,RC,C),artist_credit(Y,NA1,AC1,RC1,C1), sim(NA,NA1,S),S>=90, eq(X,X1), eq(Y,Y1), eq(AI,AI1),
artist_credit_name(X1,P,AI,NA2),artist_credit_name(Y1,P,AI1,NA3),not empty(P). % v4 added conditions


%active(X,Y):-X!=Y,artist_credit(X,NA,AC,RC,C),artist_credit(Y,NA1,AC1,RC1,C1), sim(NA,NA1,S),S>=90, eq(X,X1), eq(Y,Y1),
%artist_credit_name(X1,P,AI,NA2),artist_credit_name(Y1,P,AI1,NA3),not empty(P),  sim(NA2,NA3,S1),S1>=90. % v4 added conditions


% [ADDED]
active(X,Y):-X!=Y,track(X,RI,MI,P,N,NA,ACI,L,IDT),track(Y,RI1,MI1,P1,N1,NA1,ACI1,L,IDT1), not empty(L),sim(NA,NA1,S), S>=95.

% it is likely a merge if two track-tuples have the same position G and the same number N and the same length L and the same is_data_track and similar names NA,NA1
active(X,Y):-X!=Y,track(X,RI,MI,P,N,NA,ACI,L,IDT),track(Y,RI1,MI1,P,N,NA1,ACI1,L,IDT),not empty(P),not empty(N),not empty(L),not empty(IDT),sim(NA,NA1,S),S>=85.


% it is likely a merge if two recording-tuples have the same length L and is_video V and similar names NA,NA1
active(X,Y):-X!=Y,recording(X,NA,ACI,L,V),recording(Y,NA1,ACI1,L,V),not empty(L),not empty(V),sim(NA,NA1,S),S>=85.

% [ADDED]
% h3 - [make soft]
active(X,Y):-X!=Y,recording(X,NA,ACI,L,V),recording(Y,NA1,ACI1,L1,V), eq(ACI,ACI1),
        sim(NA,NA1,S), S>=85, not empty(V).

% it is likely a merge if two recording-tuples have the same track eq(TI,TI1) and similar names NA,NA1
active(X,Y):-X!=Y,recording(X,NA,ACI,L1,V1),recording(Y,NA1,ACI1,L2,V2),eq(X,RI),eq(Y,RI1),eq(TI,TI1),track(TI,RI,MI,P,N,NA2,ACI2,L3,IDT),track(TI1,RI1,MI1,P1,N1,NA3,ACI3,L4,IDT1),sim(NA,NA1,S),S>=85.

% it is likely a merge if two release-tuples have the same artist_credit eq(ACI,ACI1) and similar names NA,NA1
active(X,Y):-X!=Y,release(X,RGI,ACI,NA,B,ST,PA,LA,SC,Q),release(Y,RGI1,ACI1,NA1,B1,ST1,PA1,LA1,SC1,Q1),eq(ACI,ACI1),sim(NA,NA1,S),S>=90.

% really costly without sim
% [ADDED] make soft, when similarity join is invalidated for release
active(X,Y):-X!=Y,release(X,RGI,ACI,NA,B,ST,PA,LA,SC,Q),
release(Y,RGI1,ACI1,NA1,B1,ST1,PA1,LA1,SC1,Q1),eq(RGI,RGI1),eq(ACI,ACI1).

% it is likely a merge if two release_group-tuples have the same artist_credit eq(ACI,ACI1) and have similar names NA2,NA3
active(X,Y):-X!=Y,release_group(X,NA2,ACI,TY),release_group(Y,NA3,ACI1,TY1),eq(ACI,ACI1),sim(NA2,NA3,S),S>=90.


% [new added]
%active(X,Y):- X!=Y, release_group(RGI,NA2,X1,TY),release_group(RGI1,NA3,Y1,TY1), eq(RGI,RGI1), 
 %  artist_credit(X,NA,AC,RC,C),artist_credit(Y,NA1,AC1,RC2,C1), eq(X,X1), eq(Y,Y1).

% [ADDED] soft, guessing when no artist_credit and no release merge could be looked at
active(X,Y):-X!=Y,release_group(X,NA2,ACI,TY),release_group(Y,NA3,ACI1,TY), sim(NA2,NA3,S), S>=90, not empty(TY).


% it is likely a merge if two label-tuples have the same label type GT and similar names NA,NA1 
active(X,Y):-X!=Y,label(X,NA,GT,ARI,END1),label(Y,NA1,GT,ARI1,END2),not empty(GT),sim(NA,NA1,S),S>=90.

% [Modified]
% it is likely a merge if two label-tuples have very similar names NA,NA1 
active(X,Y):-X!=Y,label(X,NA,GT,ARI,END),label(Y,NA1,GT1,ARI1,END),sim(NA,NA1,S),S>=90, not empty(END).

% [ADDED]
% it is likely a merge if two label-tuples have the same area ARI,ARI1 and less similar names NA,NA1 
active(X,Y):- X!=Y, label(X,NA,GT,ARI,END1),label(Y,NA1,GT1,ARI1,END2), eq(ARI,ARI1), sim(NA,NA1,S),S>=90.

% it is likely a merge if two place-tuples have the same coordinate CD
active(X,Y):-X!=Y,place(X,NA,PT,ADD1,ARI1,CD,END),place(Y,NA1,PT1,ADD2,ARI2,CD,END1),not empty(CD).

% it is likely a merge if two place-tuples are in the same area eq(ARI1,ARI2) and have similar names NA,NA1
active(X,Y):-X!=Y,place(X,NA,PT,ADD1,ARI1,CD1,END),place(Y,NA1,PT1,ADD2,ARI2,CD2,END1),eq(ARI1,ARI2),sim(NA,NA1,S),S>=90.

%  [ADDED]
active(X,Y):-X!=Y,place(X,NA,PT,ADD1,ARI1,CD1,END),place(Y,NA1,PT,ADD2,ARI2,CD2,END1), sim(NA,NA1,S), S>=90, not empty(PT), sim(ADD1,ADD2,S2),S2>=90.

% h27
%!trace_rule{" area (%,%)-(%,%) actived since they have similar names (%,%):% and join on (area type:%, end:%)", X,ARI1,Y,ARI2,NA,NA1,S,ART,END}.
active(X,Y):-X!=Y,area(X,NA,ART,END),area(Y,NA1,ART,END),not empty(ART),not empty(END), sim(NA,NA1,S), S>=90.

% [ADDED] soft merge, area conditioned on artist, lower name similarity
%!trace_rule{" area (%,%)-(%,%) actived since they have similar names (%,%):% and join on (area type:%, end:%)", X,ARI1,Y,ARI2,NA,NA1,S,ART,END}.
active(X,Y):-X!=Y,area(X,NA,ART,END),area(Y,NA1,ART1,END1),
artist(AI,NA2,SN,BY,BM,BD,EY,EM,ED,END2,AT,G,X1,C2), sim(NA,NA1,S), S>=85,eq(X,X1), eq(Y,Y1), 
artist(AI1,NA3,SN1,BY1,BM1,BD1,EY1,EM1,ED1,END3,AT1,G1,Y1,C3),eq(AI,AI1).

% [ADDED] soft merge, area conditioned on label
%!trace_rule{" area (%,%)-(%,%) actived since they have similar names (%,%):% and join on (area type:%, end:%)", X,ARI1,Y,ARI2,NA,NA1,S,ART,END}.
%active(X,Y):-X!=Y,area(X,NA,ART,END),area(Y,NA1,ART1,END1), sim(NA,NA1,S), S>=90, 
%label(LI,NA2,GT,X1,END2),label(LI1,NA3,GT1,Y1,END3),eq(LI,LI1), eq(X,X1), eq(Y,Y1).

% [ADDED] soft merge, artist conditioned on names, when info is incomplete
active(X,Y):-X!=Y,artist(X,NA,SN,BY,BM,BD,EY,EM,ED,END,AT,G,ARI,C), empty(G), empty(C), empty(ARI), empty(BY), empty(AT),
artist(Y,NA1,SN1,BY1,BM1,BD1,EY1,EM1,ED1,END1,AT1,G1,ARI1,C1), sim(NA,NA1,S), S>=97, sim(SN,SN1,S1), S1>=97.


% reject the release merge where the releases have different barcodes if not empty
%!trace_rule{" release merge (%,%)-(%,%) is rejected since they have different barcodes (%,%)", X,REI,Y,REI1,B,B1}.
:- release(X,RGI,AC,NA,B,S,PA,LA,SC,Q), release(Y,RGI1,AC1,NA1,B1,S1,PA1,LA1,SC1,Q1), eq(X,Y), B!=B1, not empty(B), not empty(B1).

%!trace_rule{" release merge (%,%)-(%,%) is rejected since they have different release group (%,%)-(%,%)", X,REI,Y,REI1,B,B1,RG,RGI,RG1,RGI1}.
:- release(X,RGI,AC,NA,B,S,PA,LA,SC,Q), release(Y,RGI1,AC1,NA1,B1,S1,PA1,LA1,SC1,Q1), eq(X,Y),  not eq(RGI,RGI1).

:- release(X,RGI,AC,NA,B,S,PA,LA,SC,Q), release(Y,RGI1,AC1,NA1,B1,S1,PA1,LA1,SC1,Q1), eq(X,Y),  not eq(AC,AC1), not empty(AC),  not empty(AC1).


:- recording(X,NA,ACI,L,V),recording(Y,NA1,ACI1,L1,V1),eq(X,Y), not eq(ACI,ACI1), not empty(ACI),  not empty(ACI1).

:- track(X,RI,MI,P,N,NA,ACI,L,IDT),track(Y,RI1,MI1,P,N1,NA1,ACI1,L1,IDT1), eq(X,Y), not eq(ACI,ACI1), not empty(ACI),  not empty(ACI1).

:- label(X,NA,GT,ARI,END1),label(Y,NA1,GT1,ARI1,END2), eq(X,Y), not empty(ARI), not empty(ARI1), 
           not eq(ARI,ARI1).


:- release_group(X,NA2,ACI,TY),release_group(Y,NA3,ACI1,TY1), eq(X,Y), not eq(ACI,ACI1), not empty(ACI),  not empty(ACI1).

%:- artist_credit(X,ACI,NA,AC,RC,C,GID),artist_credit(Y,ACI1,NA1,AC1,RC2,C1,GID1), eq(X,Y), #count{1:sim(NA,NA1,S), S<85; 1: not sim(NA,NA1,S)}>=1.

:- artist_credit_name(ACI,P,AI,NA2),artist_credit_name(ACI1,P,AI1,NA3), not empty(P), eq(ACI,ACI1), not eq(AI,AI1).

% if we have fd says R(x) -> R'(y) and not R(x)-> R'(y')
% it implies that in the specification, if any rules depict merge from R to R', 
% it has to be complied to the DC, i.e. including conditions of R(x) -> R'(y)
% the other direction is also needed if symmetric

% reject artist_credit merge x,y for any artist in x but not in y
%:-  eq(X,Y),
 %   artist_credit_name(X,P,AI,NA4),artist_credit_name(Y,P,AI1,NA5), not eq(AI,AI1).

%:- track(X,RI,MI,P,N,NA,ACI,L,IDT),track(Y,RI1,MI1,P,N1,NA1,ACI1,L1,IDT1), eq(X,Y), not empty(P), not eq(RI,RI1).