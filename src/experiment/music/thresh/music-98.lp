%%%%%%%%%%% hard rules
% track(id, recording, medium, position, number, name, artist_credit, length, is_data_track)
% it is a merge if two track-tuples on the same position P of the same medium eq(MI,MI1), and they have similar names NA, NA1
eq(X,Y):-X!=Y,track(X,RI,MI,P,N,NA,ACI,L,IDT),track(Y,RI1,MI1,P,N1,NA1,ACI1,L1,IDT1),eq(MI,MI1),not empty(P),sim(NA,NA1,S),S>=98.

% it is a merge if two track-tuples have the same artist_credit eq(ACI,ACI1) and have similar names NA, NA1
eq(X,Y):-X!=Y,track(X,RI,MI,P,N,NA,ACI,L,IDT),track(Y,RI1,MI1,P1,N1,NA1,ACI1,L1,IDT1),eq(ACI,ACI1),sim(NA,NA1,S),S>=98.

% recording(id, name, artist_credit, length, video)
% it is a merge if two recording-tuples have the same artist_credit eq(ACI,ACI1) and similar names and the same length L and is_video V
eq(X,Y):-X!=Y,recording(X,NA,ACI,L,V),recording(Y,NA1,ACI1,L,V),eq(ACI,ACI1),sim(NA,NA1,S),S>=98,not empty(L),not empty(V).

% medium(id, release, position, format, track_count,)
% it is a merge if two medium-tuples have the same release eq(REI,REI1) and are of the same position P
eq(X,Y):-X!=Y,medium(X,REI,P,F,TC),medium(Y,REI1,P,F1,TC1),eq(REI,REI1),not empty(P).

% release(id, release_group, artist_credit, name, barcode, status, packaging, language, script, quality,)
% it is a merge if two release-tuples have the same barcode
eq(X,Y):-X!=Y,release(X,RGI,AC,NA,B,S,PA,LA,SC,Q),release(Y,RGI1,AC1,NA1,B,S1,PA1,LA1,SC1,Q1),not empty(B).

% release_group(id, name, artist_credit, type,)
% it is a merge if two release_group-tuples have the same release eq(REI,REI1) and similar names NA,NA1
eq(X,Y):-X!=Y,release_group(X,NA,AC,TY),release_group(Y,NA1,AC1,TY1),eq(X,RGI),eq(Y,RGI1),eq(REI,REI1),release(REI,RGI,AC2,NA2,B2,S2,PA2,LA2,SC2,Q2),release(REI1,RGI1,AC3,NA3,B3,S3,PA3,LA3,SC3,Q3),sim(NA,NA1,S),S>=98.

% it is a merge if two release_group-tuples have the same release eq(REI,REI1) and the same artist_credit eq(ACI,ACI1)
eq(X,Y):-X!=Y,release_group(X,NA,ACI,TY),release_group(Y,NA1,ACI1,TY1),eq(ACI,ACI1),eq(X,RGI),eq(Y,RGI1),eq(REI,REI1),release(REI,RGI,AC2,NA2,B2,S2,PA2,LA2,SC2,Q2),release(REI1,RGI1,AC3,NA3,B3,S3,PA3,LA3,SC3,Q3).

% artist(id, name, sort_name, begin_date_year, begin_date_month, begin_date_day, end_date_year, end_date_month, end_date_day, ended, artist_type, gender, area, comment, )
% it is a merge if two artist-tuples appear in the same artist_credit eq(ACI,ACI1) and are of the same position P
eq(X,Y):-X!=Y,artist(X,NA,SN,BY,BM,BD,EY,EM,ED,END,AT,G,ARI,C),artist(Y,NA1,SN1,BY1,BM1,BD1,EY1,EM1,ED1,END1,AT1,G1,ARI1,C1),eq(X,AI),eq(Y,AI1),artist_credit_name(ACI,P,AI,NA4),artist_credit_name(ACI1,P,AI1,NA5),not empty(P),eq(ACI,ACI1).


% area(id, name, area_type, ended)
% it is a merge if two area-tuples have the same area type ART, and have the same place eq(PI,PI1), and have similar names NA,NA1
eq(X,Y):-X!=Y,area(X,NA,ART,END),area(Y,NA1,ART,END1),not empty(ART),eq(X,ARI),eq(Y,ARI1),eq(PI,PI1),place(PI,NA2,PT,ADD,ARI,CD,END2),place(PI1,NA3,PT1,ADD1,ARI1,CD1,END3),sim(NA,NA1,S),S>=98.

% label(id, name, label_type, area, ended,)
% it is a merge if two label-tuples belong to the same area eq(ARI,ARI1) and have the same label type LT, and have similar names NA,NA1
eq(X,Y):-X!=Y,label(X,NA,LT,ARI,END1),label(Y,NA1,LT,ARI1,END2),eq(ARI,ARI1),not empty(LT),sim(NA,NA1,S),S>=98.

% place(id, name, place_type, address, area, coordinates, ended,)
% it is a merge if two place-tuples belong to the same area eq(ARI1,ARI2) and have the same place type PT, and have similar names NA,NA1
eq(X,Y):-X!=Y,place(X,NA,PT,ADD,ARI1,CD,END),place(Y,NA1,PT,ADD1,ARI2,CD1,END1),not empty(PT),eq(ARI1,ARI2),sim(NA,NA1,S),S>=98.

%%%%%%%%%%% soft rules
% it is likely a merge if two artist-tuples are in the same area eq(ARI,ARI1) and have simialr names NA,NA1
active(X,Y):-X!=Y,artist(X,NA,SN,BY,BM,BD,EY,EM,ED,END,AT,G,ARI,C),artist(Y,NA1,SN1,BY1,BM1,BD1,EY1,EM1,ED1,END1,AT1,G1,ARI1,C1),eq(ARI,ARI1),sim(NA,NA1,S),S>=98.

% it is likely a merge if two release-tuples belong to the same release_group eq(RGI,RGI1) and have similar names NA,NA1
active(X,Y):-X!=Y,release(X,RGI,AC,NA,B,ST,PA,LA,SC,Q),not empty(ST),release(Y,RGI1,AC1,NA1,B1,ST,PA1,LA1,SC1,Q1),eq(RGI,RGI1),sim(NA,NA1,S),S>=98.

% it is likely a merge if two release_group-tuples have the same artist_credit eq(ACI,ACI1) and have similar names NA2,NA3
active(X,Y):-X!=Y,release_group(X,NA2,ACI,TY),release_group(Y,NA3,ACI1,TY1),eq(ACI,ACI1),sim(NA2,NA3,S),S>=98.

% it is likely a merge if two artist-tuples have the same born year BY and the same artist type AT and have similar names NA,NA1,
active(X,Y):-X!=Y,artist(X,NA,SN,BY,BM,BD,EY,EM,ED,END,AT,G,ARI,C),not empty(BY),not empty(AT),artist(Y,NA1,SN1,BY,BM1,BD1,EY1,EM1,ED1,END1,AT,G1,ARI1,C1),sim(NA,NA1,S),S>=98.

% it is likely a merge if two artist-tuples have the same gender G and have similar names NA,NA1 and similar sort_name SN,SN1
active(X,Y):-X!=Y,artist(X,NA,SN,BY,BM,BD,EY,EM,ED,END,AT,G,ARI,C),not empty(G),artist(Y,NA1,SN1,BY1,BM1,BD1,EY1,EM1,ED1,END1,AT1,G,ARI1,C1),sim(NA,NA1,S),S>=98,sim(SN,SN1,S1),S1>=98.

% it is likely a merge if two track-tuples have the same position G and the same number N and the same length L and the same is_data_track and similar names NA,NA1
active(X,Y):-X!=Y,track(X,RI,MI,P,N,NA,ACI,L,IDT),track(Y,RI1,MI1,P,N,NA1,ACI1,L,IDT),not empty(P),not empty(N),not empty(L),not empty(IDT),sim(NA,NA1,S),S>=98.

% it is likely a merge if two recording-tuples have the same length L and is_video V and similar names NA,NA1
active(X,Y):-X!=Y,recording(X,NA,ACI,L,V),recording(Y,NA1,ACI1,L,V),not empty(L),not empty(V),sim(NA,NA1,S),S>=98.

% it is likely a merge if two recording-tuples have the same track eq(TI,TI1) and similar names NA,NA1
active(X,Y):-X!=Y,recording(X,NA,ACI,L1,V1),recording(Y,NA1,ACI1,L2,V2),eq(X,RI),eq(Y,RI1),eq(TI,TI1),track(TI,RI,MI,P,N,NA2,ACI2,L3,IDT),track(TI1,RI1,MI1,P1,N1,NA3,ACI3,L4,IDT1),sim(NA,NA1,S),S>=98.

% it is likely a merge if two release-tuples have the same artist_credit eq(ACI,ACI1) and similar names NA,NA1
active(X,Y):-X!=Y,release(X,RGI,ACI,NA,B,ST,PA,LA,SC,Q),release(Y,RGI1,ACI1,NA1,B1,ST1,PA1,LA1,SC1,Q1),eq(ACI,ACI1),sim(NA,NA1,S),S>=98.

% it is likely a merge if two artist_credit-tuples have the same artist_count AC and the same reference count RC and similar names and exist associated artist_credit_names with the same position P and similar names NA2,NA3 
active(X,Y):-X!=Y,artist_credit(X,NA,AC,RC,C),artist_credit(Y,NA1,AC,RC,C1),not empty(AC),not empty(RC),eq(X,ACI),eq(Y,ACI1),artist_credit_name(ACI,P,AI,NA2),artist_credit_name(ACI1,P,AI1,NA3),not empty(P),sim(NA,NA1,S),S>=98,sim(NA2,NA3,S1),S1>=98.

% it is likely a merge if two label-tuples have the same label type GT and similar names NA,NA1 
active(X,Y):-X!=Y,label(X,NA,GT,ARI,END1),label(Y,NA1,GT,ARI1,END2),not empty(GT),sim(NA,NA1,S),S>=98.

% it is likely a merge if two label-tuples have very similar names NA,NA1 
active(X,Y):-X!=Y,label(X,NA,GT,ARI,END1),label(Y,NA1,GT1,ARI1,END2),sim(NA,NA1,S),S>=98.

% it is likely a merge if two place-tuples have the same coordinate CD
active(X,Y):-X!=Y,place(X,NA,PT,ADD1,ARI1,CD,END),place(Y,NA1,PT1,ADD2,ARI2,CD,END1),not empty(CD).

% it is likely a merge if two place-tuples are in the same area eq(ARI1,ARI2) and have similar names NA,NA1
active(X,Y):-X!=Y,place(X,NA,PT,ADD1,ARI1,CD1,END),place(Y,NA1,PT1,ADD2,ARI2,CD2,END1),eq(ARI1,ARI2),sim(NA,NA1,S),S>=98.

% it is likely a merge if two area-tuples are in the same area type ART and the same end END have similar names NA,NA1
active(X,Y):-X!=Y,area(X,NA,ART,END),area(Y,NA1,ART,END),not empty(ART),not empty(END),sim(NA,NA1,S),S>=98.



%%%%%%%%%%% denials
% reject when a release merge have different barcodes if not empty
:-release(X,RGI,AC,NA,B,S,PA,LA,SC,Q),release(Y,RGI1,AC1,NA1,B1,S1,PA1,LA1,SC1,Q1),eq(X,Y),B!=B1,not empty(B),not empty(B1).
% reject when a release merge have different release_group
:- release(X,RGI,AC,NA,B,S,PA,LA,SC,Q),release(Y,RGI1,AC1,NA1,B1,S1,PA1,LA1,SC1,Q1),eq(X,Y),not eq(RGI,RGI1).  % to check this one
% reject when a label belong to different areas if not empty
:-label(X,NA,GT,ARI,END1),label(Y,NA1,GT1,ARI1,END2),eq(X,Y),not eq(ARI,ARI1),not empty(ARI),not empty(ARI1).

:- artist_credit_name(ACI,P,AI,NA2),artist_credit_name(ACI1,P,AI1,NA3), not empty(P), eq(ACI,ACI1), not eq(AI,AI1).