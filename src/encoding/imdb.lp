%% hard rules

% title_basics (X,Y) are merged because really similar primary titles PT1,PT2 , 
% original titles (OT1,OT2) and joins on (type:TT,year:YE,genre:GEN).
eq(X,Y) :- title_basics(X,TT,PT1,OT1,IA,YE,GEN), 
           %%% title_basics(title_id , 'titleType', 'primaryTitle', 'originalTitle', 'isAdult', 'startYear', 'genres') 
           %%% stores basics information of a movie (title)    
            title_basics(Y,TT,PT2,OT2,IA,YE,GEN),
            sim(PT1,PT2,S1), S1>=98,
            sim(OT1,OT2,S2), S2>=98,
            not empty(GEN),
            not empty(YE),
            not empty(IA),
            not empty(TT),
            X!=Y.


%title_basics (X,Y) are merged because similar primary titles PT1,PT2 and joins on (year:YE,genre:GEN), 
% with title alias that have similar title T1,T2 and join on region RE and order O.
eq(X,Y) :- title_basics(X,TT1,PT1,OT1,IA1,YE,GEN), 
        title_basics(Y,TT2,PT2,OT2,IA2,YE,GEN),
        sim(PT1,PT2,S1), S1>=95, 
        eq(X1,X),
        eq(Y1,Y),
        %%% title_akas(title_id, ordering, title, region, is_original_title)
        %%% stores regional title name akas of a title_basics tuple
        title_akas(X1, O, T1, RE, IO1), title_akas(Y1, O, T2, RE, IO2), 
        not empty(O),
        not empty(RE),
        sim(T1,T2,S3), S3>=95, 
        not empty(GEN),
        not empty(YE),
        X!=Y.


%title_basics (X,Y) are merged because similar primary titles PT1,PT2 and joins on genre:GEN, 
% and have the same actor (name_basics) N1,N2 on the same order O of their principal lists with the same catergory C 
eq(X,Y) :- title_basics(X,TT1,PT1,OT1,IA1,YE,GEN), 
        title_basics(Y,TT2,PT2,OT2,IA2,YE1,GEN),
        sim(PT1,PT2,S1), S1>=95, 
        eq(X,X1), eq(Y,Y1),
        %%% title_principals(title_id, 'ordering', name_basics_id, 'category')
        %%% Contains the principal cast/crew (name_basics) for titles
        title_principals(X1,O,N1,C), title_principals(Y1,O,N2,C),
        eq(N1,N2),
        not empty(GEN),
        X!=Y.
        

% name_basics (X,Y) are merged because similar primary names (PN1,PN2) and works in the same catergory C 
% and at the same order O of the principal lists of two merged titles TC1,TC2.

%%% name_basics(name_basics_id,primary_name,primary_profession)
%%% stores the information of a movie crew member
eq(X,Y) :- name_basics(X,PN1,PP1), name_basics(Y,PN2,PP2),
            sim(PN1,PN2,S2), S2>=95,  
            eq(X,X1), eq(Y,Y1),
            title_principals(TC1,O,X1,C), title_principals(TC2,O,Y1,C),
            not empty(O),
            eq(TC1,TC2), 
            X!=Y.

% name_basics (X,Y) are merged because the have exact the same primary name PN and primary profession PP
eq(X,Y):- X!=Y, name_basics(X,PN,PP), name_basics(Y,PN,PP), not empty(PN), not empty(PP).
            

%% soft rules
% s1# if names of two title_basics are reasonably simiar and released in the same year, then is possibly a merge
% title_basics X,Y are possibly a merge because of similar primary titles PT1 ,PT2 and released in the same year YE and have the same genre GEN.
active(X,Y) :- title_basics(X,TT1,PT1,OT1,IA1,YE,GEN), 
            title_basics(Y,TT2,PT2,OT2,IA2,YE,GEN),
            not empty(YE),
            not empty(GEN),
            sim(PT1,PT2,S1), S1>=90, X!=Y.


% title_basics X,Y are possibly a merge because of similar primary titles PT1 ,PT2 and released in the same year YE and have the same genre GEN.
active(X,Y) :- title_basics(X,TT1,PT1,OT1,IA1,YE,GEN), 
        title_basics(Y,TT2,PT2,OT2,IA2,YE1,GEN1),
        sim(PT1,PT2,S1), S1>=90, 
        eq(X,X1), eq(Y,Y1),
        title_principals(X1,O1,N1,C), title_principals(Y1,O2,N2,C1),
        eq(N1,N2),
        X!=Y.


% title_basics X,Y are possibly a merge because of similar primary titles PT1 ,PT2
% and similar original titles OT1 ,OT2 and have the same genre GEN with the same average rating AR.
active(X,Y) :- title_basics(X,TT1,PT1,OT1,IA1,YE1,GEN), 
            title_basics(Y,TT2,PT2,OT2,IA2,YE2,GEN),
            not empty(GEN),
            sim(PT1,PT2,S1), S1>=90, sim(OT1,OT2,S2), S2>=90, 
            eq(X,X1), eq(Y,Y1),
            %%% title_ratings(title_id', 'averageRating', 'numVotes')
            %%% Contains the IMDb rating and votes information for titles
            title_ratings( X1, AR, VO1), title_ratings(Y1, AR, VO2),not empty(AR), X!=Y.


% title_basics X,Y are possibly a merge because of similar primary titles PT1 ,PT2
% and released in the same year YE and have similar aka titles T1,T2 in the same region RE.
active(X,Y) :- title_basics(X,TT1,PT1,OT1,IA1,YE,GEN1), 
            title_basics(Y,TT2,PT2,OT2,IA2,YE,GEN2),
            not empty(YE),
            sim(PT1,PT2,S1), S1>=90, 
            % TODO, tuple id is not needed if not entities of targeting relations
            not empty(RE),
            eq(X,X1), eq(Y,Y1),
            title_akas(X1, O1, T1, RE, IO1), title_akas(Y1, O2, T2, RE, IO2), 
            sim(T1,T2,S3), S3>=90, X!=Y. %  maybe can be grouped by language


% title_basics X,Y are possibly a merge because of similar original titles OT1 ,OT2
% and released in the same year YE and have similar aka titles T1,T2 in the same region RE.
active(X,Y) :- title_basics(X,TT1,PT1,OT1,IA1,YE,GEN1), 
            title_basics(Y,TT2,PT2,OT2,IA2,YE,GEN2),
            not empty(YE),
            sim(OT1,OT2,S1), S1>=90, 
            % TODO, tuple id is not needed if not entities of targeting relations
            not empty(RE),
            eq(X,X1), eq(Y,Y1),
            title_akas(X1, O1, T1, RE, IO1), title_akas(Y1, O2, T2, RE, IO2), 
            sim(T1,T2,S3), S3>=90, X!=Y. %  maybe can be grouped by language




% title_basics X,Y are possibly a merge because of highly similar primary titles PT1, PT2
% and have the same title type TT and the same is_adult IA
active(X,Y) :- title_basics(X,TT,PT1,OT1,IA,YE1,GEN1), 
            title_basics(Y,TT,PT2,OT2,IA,YE2,GEN2), not empty(TT), not empty(IA), sim(PT1,PT2,S), S>=98.


% name basic X,Y are possibly a merge because of similar primary names PN1, PN2 and the same primary profession PP
active(X,Y) :- name_basics(X,PN1,PP), name_basics(Y,PN2,PP), not empty(PP),
           sim(PN1,PN2,S2), S2>=90, X!=Y.

% name basic X,Y are possibly a merge because of highly similar primary names PN1, PN2
active(X,Y) :- name_basics(X,PN1,PP1), name_basics(Y,PN2,PP2),
           sim(PN1,PN2,S2), S2>=98, X!=Y.


% title basics merge has to be joined on at least one of the attributes title type TT1,TT2, is_adult IA1,IA2, released year YE1,YE2, genre GEN1,GEN2
:- title_basics(X,TT1,PT1,OT1,IA1,YE,GEN), title_basics(Y,TT2,PT2,OT2,IA2,YE1,GEN1), eq(X,Y), 
        TT1!=TT2, IA1!=IA2, YE!=YE1, GEN!=GEN1.


{eq(X,Y)} :- active(X,Y).

:-title_basics(X,TT1,PT1,OT1,IA1,YE,GEN),title_basics(Y,TT2,PT2,OT2,IA2,YE1,GEN1),eq(X,Y),TT1!=TT2,IA1!=IA2,YE!=YE1,GEN!=GEN1.

eq(X,Y) :- eq(X,Z),eq(Z,Y).
eq(X,Y) :- eq(Y,X).

empty(nan). empty("nan").
#show.
#show(name_basics,V0,V3):name_basics(V0,V1,V2),name_basics(V3,V4,V5),eq(V0,V3),V0!=V3.
#show(title_basics,V0,V7):title_basics(V0,V1,V2,V3,V4,V5,V6),title_basics(V7,V8,V9,V10,V11,V12,V13),eq(V0,V7),V0!=V7.