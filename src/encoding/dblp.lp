% it is a merge if title of a dblp tuple is very similar to the title of an acm tuple T,T1, and they have the same venue V and published in the same year YE
%%% dblp(id, title, authors, venue, year), acm(id, title, authors, venue, year)
%%% stores publication records of dblp and acm respectively 
eq(X,Y) :- dblp(X,T,A1,V,YE), acm(Y,T1,A2,V,YE), sim(T,T1,S), S>=95, not empty(V), not empty(YE). 

% it is a merge if title of a dblp tuple is very similar to the title of an acm tuple T,T1, and they have very similar author names A1,A2 and published in the same year YE
eq(X,Y) :- dblp(X,T,A1,V1,YE), acm(Y,T1,A2,V2,YE), sim(T,T1,S), S>=95, sim(A1,A2,S1), S1>=95, not empty(YE). 

% it is possibly a merge if title of a dblp tuple is similar to the title of an acm tuple T,T1, and they have similar author names A1,A2 and published in the same year YE
active(X,Y):- dblp(X,T,A1,V1,Y1), acm(Y,T1,A2,V2,Y2), sim(T,T1,S), S>=90, sim(A1,A2,S1), S1>=85. 

% it is possibly a merge if title of a dblp tuple is very similar T,T1
active(X,Y):- dblp(X,T,A1,V1,Y1), acm(Y,T1,A2,V2,Y2), sim(T,T1,S), S>=95. 


% it is possibly a merge if title of a dblp tuple is similar and published in the same year YE
active(X,Y):- dblp(X,T,A1,V1,YE), acm(Y,T1,A2,V2,YE), sim(T,T1,S), S>=90, not empty(YE). 
                     
% it is possibly a merge if title of a dblp tuple is similar to the title of an acm tuple T,T1, and they have the similar venue V1,V2
active(X,Y) :- dblp(X,T,A1,V1,YE), acm(Y,T1,A2,V2,YE1), sim(T,T1,S), S>=90, sim(V1,V2,S1), S1>=90. 

% it is possibly a merge if title of a dblp tuple is very similar to the title of an acm tuple T,T1, and they have the similar authors A1,A2
active(X,Y) :- dblp(X,T,A1,V1,YE), acm(Y,T1,A2,V2,YE1), sim(T,T1,S), S>=90, sim(A1,A2,S1), S1>=90. 

% merge of dblp and acm ids must have at least one of {title:T,T1, authors:A1,A2, venue:V1,V2} similar
:- eq(X,Y),  dblp(X,T,A1,V1,Y1), acm(Y,T1,A2,V2,Y2), not sim(T,T1,_),
not sim(A1,A2,_), not sim(V1,V2,_). 

% merge of dblp and acm ids must not have both unsimilar authors A1,A2 and published in different years Y1,Y2 (make less sense in general, but effective on the dataset)
:- eq(X,Y),  dblp(X,T,A1,V1,Y1), acm(Y,T1,A2,V2,Y2),
not sim(A1,A2,_), Y1!=Y2. 



{eq(X,Y)} :- active(X,Y).

empty(nan). empty("nan").empty("ーーー").

eq(X,Y) :- eq(X,Z),eq(Z,Y).
eq(X,Y) :- eq(Y,X).
