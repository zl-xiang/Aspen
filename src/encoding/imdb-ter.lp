eq(X,Y,8):-title_basics(X,TT,PT1,OT1,IA,YE,GEN),title_basics(Y,TT,PT2,OT2,IA,YE,GEN),sim(PT1,PT2,S1),S1>=98,sim(OT1,OT2,S2),S2>=98,not empty(GEN),not empty(YE),not empty(IA),not empty(TT),X!=Y.
eq(X,Y,8):-title_basics(X,TT1,PT1,OT1,IA1,YE,GEN),title_basics(Y,TT2,PT2,OT2,IA2,YE,GEN),sim(PT1,PT2,S1),S1>=95,eq(X1,X,8),eq(Y1,Y,8),title_akas(X1,O,T1,RE,IO1),title_akas(Y1,O,T2,RE,IO2),not empty(O),not empty(RE),sim(T1,T2,S3),S3>=95,not empty(GEN),not empty(YE),X!=Y.
eq(X,Y,8):-title_basics(X,TT1,PT1,OT1,IA1,YE,GEN),title_basics(Y,TT2,PT2,OT2,IA2,YE1,GEN),sim(PT1,PT2,S1),S1>=95,eq(X,X1,8),eq(Y,Y1,8),title_principals(X1,O,N1,C),title_principals(Y1,O,N2,C),eq(N1,N2,5),not empty(GEN),X!=Y.
eq(X,Y,5):-name_basics(X,PN1,PP1),name_basics(Y,PN2,PP2),sim(PN1,PN2,S2),S2>=95,eq(X,X1,5),eq(Y,Y1,5),title_principals(TC1,O,X1,C),title_principals(TC2,O,Y1,C),not empty(O),eq(TC1,TC2,8),X!=Y.
eq(X,Y,5):-X!=Y,name_basics(X,PN,PP),name_basics(Y,PN,PP),not empty(PN),not empty(PP).
active(X,Y,8):-title_basics(X,TT1,PT1,OT1,IA1,YE,GEN),title_basics(Y,TT2,PT2,OT2,IA2,YE,GEN),not empty(YE),not empty(GEN),sim(PT1,PT2,S1),S1>=90,X!=Y.
active(X,Y,8):-title_basics(X,TT1,PT1,OT1,IA1,YE,GEN),title_basics(Y,TT2,PT2,OT2,IA2,YE1,GEN1),sim(PT1,PT2,S1),S1>=90,eq(X,X1,8),eq(Y,Y1,8),title_principals(X1,O1,N1,C),title_principals(Y1,O2,N2,C1),eq(N1,N2,5),X!=Y.
active(X,Y,8):-title_basics(X,TT1,PT1,OT1,IA1,YE1,GEN),title_basics(Y,TT2,PT2,OT2,IA2,YE2,GEN),not empty(GEN),sim(PT1,PT2,S1),S1>=90,sim(OT1,OT2,S2),S2>=90,eq(X,X1,8),eq(Y,Y1,8),title_ratings(X1,AR,VO1),title_ratings(Y1,AR,VO2),not empty(AR),X!=Y.
active(X,Y,8):-title_basics(X,TT1,PT1,OT1,IA1,YE,GEN1),title_basics(Y,TT2,PT2,OT2,IA2,YE,GEN2),not empty(YE),sim(PT1,PT2,S1),S1>=90,not empty(RE),eq(X,X1,8),eq(Y,Y1,8),title_akas(X1,O1,T1,RE,IO1),title_akas(Y1,O2,T2,RE,IO2),sim(T1,T2,S3),S3>=90,X!=Y.
active(X,Y,8):-title_basics(X,TT1,PT1,OT1,IA1,YE,GEN1),title_basics(Y,TT2,PT2,OT2,IA2,YE,GEN2),not empty(YE),sim(OT1,OT2,S1),S1>=90,not empty(RE),eq(X,X1,8),eq(Y,Y1,8),title_akas(X1,O1,T1,RE,IO1),title_akas(Y1,O2,T2,RE,IO2),sim(T1,T2,S3),S3>=90,X!=Y.
active(X,Y,8):-title_basics(X,TT,PT1,OT1,IA,YE1,GEN1),title_basics(Y,TT,PT2,OT2,IA,YE2,GEN2),not empty(TT),not empty(IA),sim(PT1,PT2,S),S>=98.
active(X,Y,5):-name_basics(X,PN1,PP),name_basics(Y,PN2,PP),not empty(PP),sim(PN1,PN2,S2),S2>=90,X!=Y.
active(X,Y,5):-name_basics(X,PN1,PP1),name_basics(Y,PN2,PP2),sim(PN1,PN2,S2),S2>=98,X!=Y.
:-title_basics(X,TT1,PT1,OT1,IA1,YE,GEN),title_basics(Y,TT2,PT2,OT2,IA2,YE1,GEN1),eq(X,Y,8),TT1!=TT2,IA1!=IA2,YE!=YE1,GEN!=GEN1.

{eq(X,Y,I)} :- active(X,Y,I).

empty(nan). empty("nan").

eq(X,Y,I) :- eq(X,Z,I),eq(Z,Y,I).
eq(X,Y,I) :- eq(Y,X,I).

#show.
#show(name_basics,V0,V3):name_basics(V0,V1,V2),name_basics(V3,V4,V5),eq(V0,V3,5),V0!=V3.
#show(title_basics,V0,V7):title_basics(V0,V1,V2,V3,V4,V5,V6),title_basics(V7,V8,V9,V10,V11,V12,V13),eq(V0,V7,8),V0!=V7.