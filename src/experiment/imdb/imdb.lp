% hard
%!trace_rule {"title_basics (%,%,%,%) are merged because really similar primary titles (%,%) , original titles (%,%) and joins on (type:%,year:%,genre:%)", X,TC1, Y,TC2, PT1, PT2, OT1,OT2, TT, YE, GEN}.
eq(X,Y) :- title_basics(X,TT,PT1,OT1,IA,YE,GEN), 
            title_basics(Y,TT,PT2,OT2,IA,YE,GEN),
            sim(PT1,PT2,S1), S1>=98,
            sim(OT1,OT2,S2), S2>=98,
            not empty(GEN),
            not empty(YE),
            not empty(IA),
            not empty(TT),
            X!=Y.

%!trace_rule {"title_basics (%,%,%,%) are merged because similar primary titles (%,%) and joins on (year:%,genre:%), with alias information that have similar title (%,%) and join on region and ordering (%,%)", X,TC1, Y,TC2, PT1, PT2, YE, GEN, T1, T2, RE,O}.
eq(X,Y) :- title_basics(X,TT1,PT1,OT1,IA1,YE,GEN), 
        title_basics(Y,TT2,PT2,OT2,IA2,YE,GEN),
        sim(PT1,PT2,S1), S1>=95, 
        eq(X1,X),
        eq(Y1,Y),
        title_akas(X1, O, T1, RE, IO1), title_akas(Y1, O, T2, RE, IO2), 
        not empty(O),
        not empty(RE),
        sim(T1,T2,S3), S3>=95, 
        not empty(GEN),
        not empty(YE),
        X!=Y.

eq(X,Y) :- title_basics(X,TT1,PT1,OT1,IA1,YE,GEN), 
        title_basics(Y,TT2,PT2,OT2,IA2,YE1,GEN),
        sim(PT1,PT2,S1), S1>=95, 
        eq(X,X1), eq(Y,Y1),
        title_principals(X1,O,N1,C), title_principals(Y1,O,N2,C),
        eq(N1,N2),
        not empty(GEN),
        X!=Y.
        

%%%%%%%%%% what's the problem here, adding this rule the procedure started to ground slow
%!trace_rule {"name_basics (%,%) are merged because similar PNs (%,%) and works in the same crew C (%) of two merged titles (%,%) ", NC1, NC2, PN1, PN2, C, TC1, TC2}.
eq(X,Y) :- name_basics(X,PN1,PP1), name_basics(Y,PN2,PP2),
            sim(PN1,PN2,S2), S2>=95,  
            eq(X,X1), eq(Y,Y1),
            title_principals(TC1,O,X1,C), title_principals(TC2,O,Y1,C),
            not empty(O),
            eq(TC1,TC2), 
            X!=Y.

eq(X,Y):- X!=Y, name_basics(X,PN,PP), name_basics(Y,PN,PP), not empty(PN), not empty(PP).
            

% soft rules
% s1# if names of two movies are reasonably simiar and released in the same year, then is possibly the same movie
%!trace_rule {"title_basics (%,%,%,%) activated because of similar primary titles (% , %) and released in the same year (%)", X,TC1,Y,TC2, PT1,PT2}.
active(X,Y) :- title_basics(X,TT1,PT1,OT1,IA1,YE,GEN), 
            title_basics(Y,TT2,PT2,OT2,IA2,YE,GEN),
            not empty(YE),
            not empty(GEN),
            sim(PT1,PT2,S1), S1>=90, X!=Y.
%-trace_rule {"title_basics (%,%) activated because of really similar original titles (% , %)", TC1,TC2, C1,C2}.
%eq(X,Y) :- title_basics(X,TC1,TT1,PT1,OT1,IA1,YE1,GEN1), 
 %           title_basics(Y,TC2,TT2,PT2,OT2,IA2,YE2,GEN2),
  %          sim(OT1,OT2,S1), S1>=98, X!=Y.


active(X,Y) :- title_basics(X,TT1,PT1,OT1,IA1,YE,GEN), 
        title_basics(Y,TT2,PT2,OT2,IA2,YE1,GEN1),
        sim(PT1,PT2,S1), S1>=90, 
        eq(X,X1), eq(Y,Y1),
        title_principals(X1,O1,N1,C), title_principals(Y1,O2,N2,C1),
        eq(N1,N2),
        X!=Y.


%!trace_rule {"title_basics (%,%) activated because of similar title (%) and (%) and joins on genre (%), same ratings (%)", TC1, TC2, PT1, PT2, GEN, AR}.
active(X,Y) :- title_basics(X,TT1,PT1,OT1,IA1,YE1,GEN), 
            title_basics(Y,TT2,PT2,OT2,IA2,YE2,GEN),
            not empty(GEN),
            sim(PT1,PT2,S1), S1>=90, sim(OT1,OT2,S2), S2>=90, 
            eq(X,X1), eq(Y,Y1),
            title_ratings( X1, AR, VO1), title_ratings(Y1, AR, VO2),not empty(AR), X!=Y.


% s2#  simlar titles and similar AKAS for the same region
% hard rules for title
%!trace_rule {"title_basics (%,%) activated because of similar title % and % , similar AKA title % and % on the same region %", TC1, TC2, PT1, PT2 T1, T2, RE}.
active(X,Y) :- title_basics(X,TT1,PT1,OT1,IA1,YE,GEN1), 
            title_basics(Y,TT2,PT2,OT2,IA2,YE,GEN2),
            not empty(YE),
            sim(PT1,PT2,S1), S1>=90, 
            % TODO, tuple id is not needed if not entities of targeting relations
            not empty(RE),
            eq(X,X1), eq(Y,Y1),
            title_akas(X1, O1, T1, RE, IO1), title_akas(Y1, O2, T2, RE, IO2), 
            sim(T1,T2,S3), S3>=90, X!=Y. %  maybe can be grouped by language

active(X,Y) :- title_basics(X,TT1,PT1,OT1,IA1,YE,GEN1), 
            title_basics(Y,TT2,PT2,OT2,IA2,YE,GEN2),
            not empty(YE),
            sim(OT1,OT2,S1), S1>=90, 
            % TODO, tuple id is not needed if not entities of targeting relations
            not empty(RE),
            eq(X,X1), eq(Y,Y1),
            title_akas(X1, O1, T1, RE, IO1), title_akas(Y1, O2, T2, RE, IO2), 
            sim(T1,T2,S3), S3>=90, X!=Y. %  maybe can be grouped by language




% added title merge
active(X,Y) :- title_basics(X,TT,PT1,OT1,IA,YE1,GEN1), 
            title_basics(Y,TT,PT2,OT2,IA,YE2,GEN2), not empty(TT), not empty(IA), sim(PT1,PT2,S), S>=98.

% s5# name basic. similar names, same primary professions
%!trace_rule {"name_basics (%,%) activated because similar name (%, %) and employed in the same profession (%)", NC1,NC2 , PN1,PN2, PP}.
active(X,Y) :- name_basics(X,PN1,PP), name_basics(Y,PN2,PP), not empty(PP),
           sim(PN1,PN2,S2), S2>=90, X!=Y.

active(X,Y) :- name_basics(X,PN1,PP1), name_basics(Y,PN2,PP2),
           sim(PN1,PN2,S2), S2>=98, X!=Y.


%:- name_basics(X,PN1,PP1), name_basics(Y,PN2,PP2), eq(X,Y), eq(X,X1),  eq(Y,Y1),
 %  title_principals(T3,O,X1,C),title_principals(T4,O,Y1,C1), eq(T3,T4), C1!=C .

%:- title_basics(X,TT1,PT1,OT1,IA1,YE,GEN), title_basics(Y,TT2,PT2,OT2,IA2,YE1,GEN1),
 %      eq(X,Y), eq(X,X1), eq(Y,Y1),title_principals(X1,O,N1,C),title_principals(Y1,O,N2,C), not eq(N1,N2),eq(N1,N1),eq(N2,N2).
% tb merge has to be joined on at least one of the attributes
:- title_basics(X,TT1,PT1,OT1,IA1,YE,GEN), title_basics(Y,TT2,PT2,OT2,IA2,YE1,GEN1), eq(X,Y), 
        TT1!=TT2, IA1!=IA2, YE!=YE1, GEN!=GEN1.

%:- name_basics(X,PN1,PP1), name_basics(Y,PN2,PP2), sim(PN1,PN2,S), S<98, PP1!=PP2.