% cora(address ADD, authors AN, booktitle BT, date D, editor ED, id X, institution INS, journal J, month M, note N, pages PA, publisher P, tech TEC, title T, type TY, volume V, year YE)

%% Note that there is no schema for cora dataset, hence exact semantics of attributes are unknown. Hence all rules are based on observations on values/ground truth, and tuned via trial and error.
%% Values of attributes (intuitively) of type date or number like year, date, pages also participate in similarity atoms because 
%% they all appear as strings that have different representations, e.g. year 1999 could appear as 19(99) or 199,9

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The following hard rules can be expressed all together with: 

%% eq(X,Y) :- X!=Y,cora(ADD, AN, BT, D,ED,X, INS,J,M, N,PA,P,TEC,T,TY,V,YE),
%%    cora( ADD1, AN1, BT1, D1,ED1,Y, INS1,J1,M1,N1,PA1,P1,TEC,T1,TY1,V1,YE1), 
%%    sim(T,T1,S1), S1>=95, 
%%    #count{ 1:sim(J,J1,S2),S2>=95; 
%%    1:sim(BT,BT1,S2),S2>=95; 
%%    1:sim(AN,AN1,S2),S2>=95} = 1,
%%    #count{1:YE=YE1,not empty(YE); 
%%    1:D=D1,not empty(D); 
%%    1:ADD=ADD1,not empty(ADD); 
%%    1:PA=PA1,not empty(PA); 
%%    1:V=V1,not empty(V); 
%%    1:sim(ED,ED1,S3),S3>=95;
%%    1:sim(P,P1,S3),S3>=95
%%    } = 1,
%%    empty(TEC).

%% That is it is a merge of cora publication id X,Y, if they have very similar titles T1,T2, and 
%% have very similar values in one of the attributes 
%%% {journal name J,J1, 
%%% booketitle BT,BT1, 
%%% author names AN,AN1} and 
%%% satisfy one of the conditions 
%%% {published in the same year YE=YE1, 
%%% published in the same date D=D1, 
%%% have the same address ADD=ADD1,
%%% on the same number of pages PA=PA1,
%%% in the same volume V=V1,
%%% have very similar editor ED,ED1,
%%% have very similar publisher P,P1}  
%% and have empty technical report TEC.


% it is a merge of cora publication id X,Y, if they have very similar titles T1,T2, and very similar journal name J1,J2 and 
% published in the same year YE and have empty technical report TEC
eq(X,Y) :- X!=Y,cora(ADD, AN, BT, D,ED,X, INS,J,M, N,PA,P,TEC,T,TY,V,YE),
      cora( ADD1, AN1, BT1, D1,ED1,Y, INS1,J1,M1,N1,PA1,P1,TEC,T1,TY1,V1,YE), 
       sim(T,T1,S1), S1>=95, sim(J,J1,S2),S2>=95,
       not empty(YE), empty(TEC).

% it is a merge of cora publication id X,Y, if they have very similar titles T1,T2, and very similar book titles BT1,BT2 and 
% published in the same year YE and have empty technical report TEC
eq(X,Y) :- X!=Y,cora(ADD, AN, BT, D,ED,X, INS,J,M, N,PA,P,TEC,T,TY,V,YE),
      cora( ADD1, AN1, BT1, D1,ED1,Y, INS1,J1,M1,N1,PA1,P1,TEC,T1,TY1,V1,YE), 
       sim(T,T1,S1), S1>=95, sim(BT,BT1,S2),S2>=95,
     not empty(YE), empty(TEC).

% it is a merge of cora publication id X,Y, if they have very similar titles T1,T2, and very similar author names AN1,AN2 and 
% published in the same year YE and have empty technical report TEC
eq(X,Y) :- X!=Y,cora(ADD, AN, BT, D,ED,X, INS,J,M, N,PA,P,TEC,T,TY,V,YE),
      cora( ADD1, AN1, BT1, D1,ED1,Y, INS1,J1,M1,N1,PA1,P1,TEC,T1,TY1,V1,YE), 
       sim(T,T1,S1), S1>=95, sim(AN,AN1,S2), S2>=95,
       not empty(YE),empty(TEC).

% it is a merge of cora publication id X,Y, if they have very similar titles T1,T2, and very similar journal name J1,J2 and 
% published in the same date D and have empty technical report TEC
eq(X,Y) :- X!=Y,cora(ADD, AN, BT, D,ED,X, INS,J,M, N,PA,P,TEC,T,TY,V,YE),
      cora( ADD1, AN1, BT1, D,ED1,Y, INS1,J1,M1,N1,PA1,P1,TEC,T1,TY1,V1,YE1), 
       sim(T,T1,S1), S1>=95, sim(J,J1,S2),S2>=95,
      not empty(D),empty(TEC).

% it is a merge of cora publication id X,Y, if they have very similar titles T1,T2, and very similar book title name BT1,BT2 and 
% published in the same date D and have empty technical report TEC
eq(X,Y) :- X!=Y,cora(ADD, AN, BT, D,ED,X, INS,J,M, N,PA,P,TEC,T,TY,V,YE),
      cora( ADD1, AN1, BT1, D,ED1,Y, INS1,J1,M1,N1,PA1,P1,TEC,T1,TY1,V1,YE1), 
       sim(T,T1,S1), S1>=95, sim(BT,BT1,S2),S2>=95,
      not empty(D),empty(TEC).

% it is a merge of cora publication id X,Y, if they have very similar titles T1,T2, and very similar author names AN1,AN2 and 
% published in the same date D and have empty technical report TEC
eq(X,Y) :- X!=Y,cora(ADD, AN, BT, D,ED,X, INS,J,M, N,PA,P,TEC,T,TY,V,YE),
       cora( ADD1, AN1, BT1, D,ED1,Y, INS1,J1,M1,N1,PA1,P1,TEC,T1,TY1,V1,YE1), 
       sim(T,T1,S1), S1>=95, sim(AN,AN1,S2), S2>=95,
       not empty(D),empty(TEC).


% it is a merge of cora publication id X,Y, if they have very similar titles T1,T2, and very similar journal names J1,J2 and 
% have the same address ADD and have empty technical report TEC
eq(X,Y) :- X!=Y,cora(ADD, AN, BT, D,ED,X, INS,J,M, N,PA,P,TEC,T,TY,V,YE),
      cora( ADD, AN1, BT1, D1,ED1,Y, INS1,J1,M1,N1,PA1,P1,TEC,T1,TY1,V1,YE1), 
       sim(T,T1,S1), S1>=95, sim(J,J1,S2),S2>=95,
       not empty(ADD),empty(TEC).

% it is a merge of cora publication id X,Y, if they have very similar titles T1,T2, and very similar booktitle names BT1,BT2 and 
% have the same address ADD and have empty technical report TEC
eq(X,Y) :- X!=Y,cora(ADD, AN, BT, D,ED,X, INS,J,M, N,PA,P,TEC,T,TY,V,YE),
      cora( ADD, AN1, BT1, D1,ED1,Y, INS1,J1,M1,N1,PA1,P1,TEC,T1,TY1,V1,YE1), 
       sim(T,T1,S1), S1>=95, sim(BT,BT1,S2),S2>=95,
       not empty(ADD),empty(TEC).

% it is a merge of cora publication id X,Y, if they have very similar titles T1,T2, and very similar author names AN1,AN2 and 
% have the same address ADD and have empty technical report TEC
eq(X,Y) :- X!=Y,cora(ADD, AN, BT, D,ED,X, INS,J,M, N,PA,P,TEC,T,TY,V,YE),
      cora( ADD, AN1, BT1, D1,ED1,Y, INS1,J1,M1,N1,PA1,P1,TEC,T1,TY1,V1,YE1), 
       sim(T,T1,S1), S1>=95, sim(AN,AN1,S2), S2>=95,
       not empty(ADD),empty(TEC).

% it is a merge of cora publication id X,Y, if they have very similar titles T1,T2, and very similar journal names J1,J2 and 
% have the same pages PA and have empty technical report TEC
eq(X,Y) :- X!=Y,cora(ADD, AN, BT, D,ED,X, INS,J,M, N,PA,P,TEC,T,TY,V,YE),
      cora( ADD1, AN1, BT1, D1,ED1,Y, INS1,J1,M1,N1,PA,P1,TEC,T1,TY1,V1,YE1), 
       sim(T,T1,S1), S1>=95, sim(J,J1,S2),S2>=95,
      not empty(PA),empty(TEC).

% it is a merge of cora publication id X,Y, if they have very similar titles T1,T2, and very similar booktitle names BT1,BT2 and 
% have the same pages PA and have empty technical report TEC
eq(X,Y) :- X!=Y,cora(ADD, AN, BT, D,ED,X, INS,J,M, N,PA,P,TEC,T,TY,V,YE),
      cora( ADD1, AN1, BT1, D1,ED1,Y, INS1,J1,M1,N1,PA,P1,TEC,T1,TY1,V1,YE1), 
       sim(T,T1,S1), S1>=95, sim(BT,BT1,S2),S2>=95,
       not empty(PA),empty(TEC).

% it is a merge of cora publication id X,Y, if they have very similar titles T1,T2, and very similar author names AN1,AN2 and 
% have the same pages PA and have empty technical report TEC
eq(X,Y) :- X!=Y,cora(ADD, AN, BT, D,ED,X, INS,J,M, N,PA,P,TEC,T,TY,V,YE),
      cora( ADD1, AN1, BT1, D1,ED1,Y, INS1,J1,M1,N1,PA,P1,TEC,T1,TY1,V1,YE1), 
       sim(T,T1,S1), S1>=95, sim(AN,AN1,S2), S2>=95,
       not empty(PA),empty(TEC).

% it is a merge of cora publication id X,Y, if they have very similar titles T1,T2, and very similar journal names J1,J2 and 
% in the same volume V and have empty technical report TEC
eq(X,Y) :- X!=Y,cora(ADD, AN, BT, D,ED,X, INS,J,M, N,PA,P,TEC,T,TY,V,YE),
      cora( ADD1, AN1, BT1, D1,ED1,Y, INS1,J1,M1,N1,PA1,P1,TEC,T1,TY1,V,YE1), 
       sim(T,T1,S1), S1>=95, sim(J,J1,S2),S2>=95,
       not empty(V),empty(TEC).

% it is a merge of cora publication id X,Y, if they have very similar titles T1,T2, and very similar booktitle names BT1,BT2 and 
% in the same volume V and have empty technical report TEC
eq(X,Y) :- X!=Y,cora(ADD, AN, BT, D,ED,X, INS,J,M, N,PA,P,TEC,T,TY,V,YE),
      cora( ADD1, AN1, BT1, D1,ED1,Y, INS1,J1,M1,N1,PA1,P1,TEC,T1,TY1,V,YE1), 
       sim(T,T1,S1), S1>=95, sim(BT,BT1,S2),S2>=95,
       not empty(V),empty(TEC).

% it is a merge of cora publication id X,Y, if they have very similar titles T1,T2, and very similar author names AN1,AN2 and 
% in the same volume V and have empty technical report TEC
eq(X,Y) :- X!=Y,cora(ADD, AN, BT, D,ED,X, INS,J,M, N,PA,P,TEC,T,TY,V,YE),
      cora( ADD1, AN1, BT1, D1,ED1,Y, INS1,J1,M1,N1,PA1,P1,TEC,T1,TY1,V,YE1), 
       sim(T,T1,S1), S1>=95, sim(AN,AN1,S2), S2>=95,
       not empty(V),empty(TEC).

       

% it is a merge of cora publication id X,Y, if they have very similar titles T1,T2, and very similar journal names J1,J2 and 
% very similar editors ED1,ED2 and have empty technical report TEC
eq(X,Y) :- X!=Y,cora(ADD, AN, BT, D,ED,X, INS,J,M, N,PA,P,TEC,T,TY,V,YE),
      cora( ADD1, AN1, BT1, D1,ED1,Y, INS1,J1,M1,N1,PA1,P1,TEC,T1,TY1,V1,YE1), 
       sim(T,T1,S1), S1>=95, sim(J,J1,S2),S2>=95,
       sim(ED,ED1,S3),S3>=95,empty(TEC).

% it is a merge of cora publication id X,Y, if they have very similar titles T1,T2, and very similar booktitle names BT1,BT2 and 
% very similar editors ED1,ED2 and have empty technical report TEC
eq(X,Y) :- X!=Y,cora(ADD, AN, BT, D,ED,X, INS,J,M, N,PA,P,TEC,T,TY,V,YE),
      cora( ADD1, AN1, BT1, D1,ED1,Y, INS1,J1,M1,N1,PA1,P1,TEC,T1,TY1,V1,YE1), 
       sim(T,T1,S1), S1>=95, sim(BT,BT1,S2),S2>=95,
       sim(ED,ED1,S3),S3>=95,empty(TEC).


% it is a merge of cora publication id X,Y, if they have very similar titles T1,T2, and very similar author names AN1,AN2 and 
% very similar editors ED1,ED2 and have empty technical report TEC
eq(X,Y) :- X!=Y,cora(ADD, AN, BT, D,ED,X, INS,J,M, N,PA,P,TEC,T,TY,V,YE),
      cora( ADD1, AN1, BT1, D1,ED1,Y, INS1,J1,M1,N1,PA1,P1,TEC,T1,TY1,V1,YE1), 
       sim(T,T1,S1), S1>=95, sim(AN,AN1,S2), S2>=95,
       sim(ED,ED1,S3),S3>=95,empty(TEC).

%----
% it is a merge of cora publication id X,Y, if they have very similar titles T1,T2, and very similar journal names J1,J2 and 
% very similar publishers P1,P2 and have empty technical report TEC
eq(X,Y) :- X!=Y,cora(ADD, AN, BT, D,ED,X, INS,J,M, N,PA,P,TEC,T,TY,V,YE),
      cora( ADD1, AN1, BT1, D1,ED1,Y, INS1,J1,M1,N1,PA1,P1,TEC,T1,TY1,V1,YE1), 
       sim(T,T1,S1), S1>=95, sim(J,J1,S2),S2>=95,
       sim(P,P1,S3),S3>=95, empty(TEC).


% it is a merge of cora publication id X,Y, if they have very similar titles T1,T2, and very similar booktitle names BT1,BT2 and 
% very similar publishers P1,P2 and have empty technical report TEC
eq(X,Y) :- X!=Y,cora(ADD, AN, BT, D,ED,X, INS,J,M, N,PA,P,TEC,T,TY,V,YE),
      cora( ADD1, AN1, BT1, D1,ED1,Y, INS1,J1,M1,N1,PA1,P1,TEC,T1,TY1,V1,YE1), 
       sim(T,T1,S1), S1>=95, sim(BT,BT1,S2),S2>=95,
       sim(P,P1,S3),S3>=95,empty(TEC).


% it is a merge of cora publication id X,Y, if they have very similar titles T1,T2, and very similar author names AN1,AN2 and  
% very similar publishers P1,P2 and have empty technical report TEC
eq(X,Y) :- X!=Y,cora(ADD, AN, BT, D,ED,X, INS,J,M, N,PA,P,TEC,T,TY,V,YE),
      cora( ADD1, AN1, BT1, D1,ED1,Y, INS1,J1,M1,N1,PA1,P1,TEC,T1,TY1,V1,YE1), 
       sim(T,T1,S1), S1>=95, sim(AN,AN1,S2), S2>=95,
       sim(P,P1,S3),S3>=95,empty(TEC).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% soft


% it is possibly a merge of cora publication id X,Y, if they have similar titles T1,T2 and similar technical report TEC
active(X,Y) :- X!=Y,cora(ADD, AN, BT, D,ED,X, INS,J,M, N,PA,P,TEC,T,TY,V,YE),
      cora( ADD1, AN1, BT1, D1,ED1,Y, INS1,J1,M1,N1,PA1,P1,TEC1,T1,TY1,V1,YE1), 
       sim(T,T1,S1), S1>=90, sim(TEC,TEC1,S2),S2>=90.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The following soft rules can be expressed all together with: 

%% active(X,Y) :- X!=Y,cora(ADD, AN, BT, D,ED,X, INS,J,M, N,PA,P,TEC,T,TY,V,YE),
%%    cora( ADD1, AN1, BT1, D1,ED1,Y, INS1,J1,M1,N1,PA1,P1,TEC,T1,TY1,V1,YE1), 
%%    sim(T,T1,S1), S1>=90, 
%%    #count{ 1:sim(J,J1,S2),S2>=85; 
%%    1:sim(BT,BT1,S2),S2>=85; 
%%    1:sim(AN,AN1,S2),S2>=85} = 1,
%%    #count{1:sim(YE,YE1,S2),S2>=95; 
%%    1:sim(D,D1,S2),S2>=95
%%    } = 1.

%% That is it possibly is a merge of cora publication id X,Y, if they have similar titles T1,T2, and 
%% have less similar values in one of the attributes 
%%% {journal name J,J1, 
%%% booketitle BT,BT1, 
%%% author names AN,AN1} and 
%%% satisfy one of the conditions 
%%% {published in the very similar years YE,YE1, 
%%% published in the very similar date D,D1}

% it is possibly a merge of cora publication id X,Y, if they have  similar titles T1,T2, and similar journal names J1,J2 and 
% published in very similar years YE,YE1 
active(X,Y) :- X!=Y,cora(ADD, AN, BT, D,ED,X, INS,J,M, N,PA,P,TEC,T,TY,V,YE),
      cora( ADD1, AN1, BT1, D1,ED1,Y, INS1,J1,M1,N1,PA1,P1,TEC1,T1,TY1,V1,YE1), 
       sim(T,T1,S1), S1>=90, sim(J,J1,S2),S2>=85,
       sim(YE,YE1,S3),S3>=95.

% it is possibly a merge of cora publication id X,Y, if they have  similar titles T1,T2, and similar booktitle names BT1,BT2 and 
% published in very similar years YE,YE1 
active(X,Y) :- X!=Y,cora(ADD, AN, BT, D,ED,X, INS,J,M, N,PA,P,TEC,T,TY,V,YE),
      cora( ADD1, AN1, BT1, D1,ED1,Y, INS1,J1,M1,N1,PA1,P1,TEC1,T1,TY1,V1,YE1), 
       sim(T,T1,S1), S1>=90, sim(BT,BT1,S2),S2>=85,
       sim(YE,YE1,S3),S3>=95.

% it is possibly a merge of cora publication id X,Y, if they have  similar titles T1,T2, and less similar author names AN1,AN2 and  
% published in very similar years YE,YE1 
active(X,Y) :- X!=Y,cora(ADD, AN, BT, D,ED,X, INS,J,M, N,PA,P,TEC,T,TY,V,YE),
      cora( ADD1, AN1, BT1, D1,ED1,Y, INS1,J1,M1,N1,PA1,P1,TEC1,T1,TY1,V1,YE1), 
       sim(T,T1,S1), S1>=90, sim(AN,AN1,S2), S2>=85,
       sim(YE,YE1,S3),S3>=95.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% The following soft rules can be expressed all together with: 

%% active(X,Y) :- X!=Y,cora(ADD, AN, BT, D,ED,X, INS,J,M, N,PA,P,TEC,T,TY,V,YE),
%%    cora( ADD1, AN1, BT1, D1,ED1,Y, INS1,J1,M1,N1,PA1,P1,TEC,T1,TY1,V1,YE1), 
%%    sim(T,T1,S1), S1>=85, 
%%    #count{ 1:sim(J,J1,S2),S2>=95; 
%%    1:sim(BT,BT1,S2),S2>=95; 
%%    1:sim(AN,AN1,S2),S2>=90} = 1,
%%    #count{1:sim(YE,YE1,S2),S2>=95; 
%%    1:sim(D,D1,S2),S2>=95
%%    } = 1.

%% That is, it possibly is a merge of cora publication id X,Y, if they have less similar titles T1,T2, and 
%%  satisfy one of the conditions 
%%% {very similar journal name J,J1, 
%%% very similar booketitle BT,BT1, 
%%% similar author names AN,AN1} and 
%%% satisfy one of the conditions 
%%% {published in the very similar years YE,YE1, 
%%% published in the very similar date D,D1}



% it is possibly a merge of cora publication id X,Y, if they have less similar titles T1,T2, and similar author names AN1,AN2 and  
% published in very similar years YE,YE1 
active(X,Y) :- X!=Y,cora(ADD, AN, BT, D,ED,X, INS,J,M, N,PA,P,TEC,T,TY,V,YE),
      cora( ADD1, AN1, BT1, D1,ED1,Y, INS1,J1,M1,N1,PA1,P1,TEC1,T1,TY1,V1,YE1), 
       sim(T,T1,S1), S1>=85, sim(AN,AN1,S2), S2>=90,
       sim(YE,YE1,S3),S3>=95.     

% it is possibly a merge of cora publication id X,Y, if they have less similar titles T1,T2, and similar author names AN1,AN2 and  
% published in very similar date D,D1 
active(X,Y) :- X!=Y,cora(ADD, AN, BT, D,ED,X, INS,J,M, N,PA,P,TEC,T,TY,V,YE),
      cora( ADD1, AN1, BT1, D1,ED1,Y, INS1,J1,M1,N1,PA1,P1,TEC1,T1,TY1,V1,YE1), 
       sim(T,T1,S1), S1>=85, sim(AN,AN1,S2), S2>=90,
       sim(D,D1,S3),S3>=95.         

% it is possibly a merge of cora publication id X,Y, if they have less similar titles T1,T2, and very similar journal names J1,J2 and 
% published in very similar date D,D1 
active(X,Y) :- X!=Y,cora(ADD, AN, BT, D,ED,X, INS,J,M, N,PA,P,TEC,T,TY,V,YE),
      cora( ADD1, AN1, BT1, D1,ED1,Y, INS1,J1,M1,N1,PA1,P1,TEC1,T1,TY1,V1,YE1), 
       sim(T,T1,S1), S1>=85, sim(J,J1,S2),S2>=95,
       sim(D,D1,S3),S3>=95. 

% it is possibly a merge of cora publication id X,Y, if they have less similar titles T1,T2, and very similar booktitle names BT1,BT2 and 
% published in very similar years YE,YE1 
active(X,Y) :- X!=Y,cora(ADD, AN, BT, D,ED,X, INS,J,M, N,PA,P,TEC,T,TY,V,YE),
      cora( ADD1, AN1, BT1, D1,ED1,Y, INS1,J1,M1,N1,PA1,P1,TEC1,T1,TY1,V1,YE1), 
       sim(T,T1,S1), S1>=85, sim(BT,BT1,S2),S2>=95,
       sim(YE,YE1,S3),S3>=95. 

% it is possibly a merge of cora publication id X,Y, if they have less similar titles T1,T2, and very similar booktitle names BT1,BT2 and 
% published in very similar date D,D1 
active(X,Y) :- X!=Y,cora(ADD, AN, BT, D,ED,X, INS,J,M, N,PA,P,TEC,T,TY,V,YE),
      cora( ADD1, AN1, BT1, D1,ED1,Y, INS1,J1,M1,N1,PA1,P1,TEC1,T1,TY1,V1,YE1), 
       sim(T,T1,S1), S1>=85, sim(BT,BT1,S2),S2>=95,
       sim(D,D1,S3),S3>=95. 

% it is possibly a merge of cora publication id X,Y, if they have similar titles T1,T2, and less similar  journal names J1,J2 and 
% published in very similar date D,D1 
active(X,Y) :- X!=Y,cora(ADD, AN, BT, D,ED,X, INS,J,M, N,PA,P,TEC,T,TY,V,YE),
      cora( ADD1, AN1, BT1, D1,ED1,Y, INS1,J1,M1,N1,PA1,P1,TEC1,T1,TY1,V1,YE1), 
       sim(T,T1,S1), S1>=90, sim(J,J1,S2),S2>=85,
       sim(D,D1,S3),S3>=95.

% it is possibly a merge of cora publication id X,Y, if they have similar titles T1,T2, and less similar booktitle names BT1,BT2 and 
% published in very similar date D,D1 
active(X,Y) :- X!=Y,cora(ADD, AN, BT, D,ED,X, INS,J,M, N,PA,P,TEC,T,TY,V,YE),
      cora( ADD1, AN1, BT1, D1,ED1,Y, INS1,J1,M1,N1,PA1,P1,TEC1,T1,TY1,V1,YE1), 
       sim(T,T1,S1), S1>=90, sim(BT,BT1,S2),S2>=85,
       sim(D,D1,S3),S3>=95.


% it is possibly a merge of cora publication id X,Y, if they have similar titles T1,T2, and less similar author names AN1,AN2 and 
% published in very similar date D,D1 
active(X,Y) :- X!=Y,cora(ADD, AN, BT, D,ED,X, INS,J,M, N,PA,P,TEC,T,TY,V,YE),
      cora( ADD1, AN1, BT1, D1,ED1,Y, INS1,J1,M1,N1,PA1,P1,TEC1,T1,TY1,V1,YE1), 
       sim(T,T1,S1), S1>=90, sim(AN,AN1,S2), S2>=85,
       sim(D,D1,S3),S3>=95.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% The following soft rules can be expressed all together with: 

%% active(X,Y) :- X!=Y,cora(ADD, AN, BT, D,ED,X, INS,J,M, N,PA,P,TEC,T,TY,V,YE),
%%    cora( ADD1, AN1, BT1, D1,ED1,Y, INS1,J1,M1,N1,PA1,P1,TEC,T1,TY1,V1,YE1), 
%%    sim(T,T1,S1), S1>=90, 
%%    #count{ 1:sim(J,J1,S2),S2>=85; 
%%    1:sim(BT,BT1,S2),S2>=85; 
%%    1:sim(AN,AN1,S2),S2>=85} = 1,
%%    #count{1: sim(ADD,ADD1,S3),S3>=90;
%%    1:sim(PA,PA1,S3),S3>=95;
%%    1:sim(V,V1,S3),S3>=95;
%%    1:sim(ED,ED1,S3),S3>=80;
%%    1:sim(P,P1,S3),S3>=80
%%    } = 1.

%% That is, it possibly is a merge of cora publication id X,Y, if they have similar titles T1,T2, and 
%%  satisfy one of the conditions 
%%% {less similar journal name J,J1, 
%%% less similar booketitle BT,BT1, 
%%% less similar author names AN,AN1} and 
%%% satisfy one of the conditions 
%%% {have similar address ADD,ADD1,
%%% have very similar number of pages PA,PA1,
%%% have similar volumes V,V1,
%%% have less similar editor ED,ED1,
%%% have less similar publisher P,P1}  


% it is possibly a merge of cora publication id X,Y, if they have similar titles T1,T2, and less similar journal names J1,J2 and 
% have similar adresses ADD,ADD1 
active(X,Y) :- X!=Y,cora(ADD, AN, BT, D,ED,X, INS,J,M, N,PA,P,TEC,T,TY,V,YE),
      cora( ADD1, AN1, BT1, D1,ED1,Y, INS1,J1,M1,N1,PA1,P1,TEC1,T1,TY1,V1,YE1), 
       sim(T,T1,S1), S1>=90, sim(J,J1,S2),S2>=85,
       sim(ADD,ADD1,S3),S3>=90.

% it is possibly a merge of cora publication id X,Y, if they have similar titles T1,T2, and less similar booktitle names BT1,BT2 and 
% have similar adresses ADD,ADD1 
active(X,Y) :- X!=Y,cora(ADD, AN, BT, D,ED,X, INS,J,M, N,PA,P,TEC,T,TY,V,YE),
      cora( ADD1, AN1, BT1, D1,ED1,Y, INS1,J1,M1,N1,PA1,P1,TEC1,T1,TY1,V1,YE1), 
           sim(T,T1,S1), S1>=90, sim(BT,BT1,S2),S2>=85,
         sim(ADD,ADD1,S3),S3>=90.

% it is possibly a merge of cora publication id X,Y, if they have similar titles T1,T2, and less similar author names AN1,AN2 and 
% have similar adresses ADD,ADD1 
active(X,Y) :- X!=Y,cora(ADD, AN, BT, D,ED,X, INS,J,M, N,PA,P,TEC,T,TY,V,YE),
      cora( ADD1, AN1, BT1, D1,ED1,Y, INS1,J1,M1,N1,PA1,P1,TEC1,T1,TY1,V1,YE1), 
       sim(T,T1,S1), S1>=90, sim(AN,AN1,S2), S2>=85,
      sim(ADD,ADD1,S3),S3>=90.

% it is possibly a merge of cora publication id X,Y, if they have similar titles T1,T2, and less similar journal names J1,J2 and 
% have very similar page count PA,PA1 
active(X,Y) :- X!=Y,cora(ADD, AN, BT, D,ED,X, INS,J,M, N,PA,P,TEC,T,TY,V,YE),
      cora( ADD1, AN1, BT1, D1,ED1,Y, INS1,J1,M1,N1,PA1,P1,TEC1,T1,TY1,V1,YE1), 
       sim(T,T1,S1), S1>=90, sim(J,J1,S2),S2>=85,
       sim(PA,PA1,S3),S3>=95.

% it is possibly a merge of cora publication id X,Y, if they have similar titles T1,T2, and less similar booktitle names BT1,BT2 and 
% have very similar page count PA,PA1 
active(X,Y) :- X!=Y,cora(ADD, AN, BT, D,ED,X, INS,J,M, N,PA,P,TEC,T,TY,V,YE),
      cora( ADD1, AN1, BT1, D1,ED1,Y, INS1,J1,M1,N1,PA1,P1,TEC1,T1,TY1,V1,YE1), 
           sim(T,T1,S1), S1>=90,sim(BT,BT1,S2),S2>=85,
  sim(PA,PA1,S3),S3>=95.

% it is possibly a merge of cora publication id X,Y, if they have similar titles T1,T2, and less similar author names AN1,AN2 and 
% have very similar page count PA,PA1 
active(X,Y) :- X!=Y,cora(ADD, AN, BT, D,ED,X, INS,J,M, N,PA,P,TEC,T,TY,V,YE),
      cora( ADD1, AN1, BT1, D1,ED1,Y, INS1,J1,M1,N1,PA1,P1,TEC1,T1,TY1,V1,YE1), 
       sim(T,T1,S1), S1>=90, sim(AN,AN1,S2), S2>=85,
  sim(PA,PA1,S3),S3>=95.

% it is possibly a merge of cora publication id X,Y, if they have similar titles T1,T2, and less similar journal names J1,J2 and 
% have very similar volume V,V1 
active(X,Y) :- X!=Y,cora(ADD, AN, BT, D,ED,X, INS,J,M, N,PA,P,TEC,T,TY,V,YE),
      cora( ADD1, AN1, BT1, D1,ED1,Y, INS1,J1,M1,N1,PA1,P1,TEC1,T1,TY1,V1,YE1), 
       sim(T,T1,S1), S1>=90, sim(J,J1,S2),S2>=85,
       sim(V,V1,S3),S3>=95.
       
% it is possibly a merge of cora publication id X,Y, if they have similar titles T1,T2, and less similar booktitle names BT1,BT2 and 
% have very similar volume V,V1 
active(X,Y) :- X!=Y,cora(ADD, AN, BT, D,ED,X, INS,J,M, N,PA,P,TEC,T,TY,V,YE),
      cora( ADD1, AN1, BT1, D1,ED1,Y, INS1,J1,M1,N1,PA1,P1,TEC1,T1,TY1,V1,YE1), 
           sim(T,T1,S1), S1>=90,sim(BT,BT1,S2),S2>=85,
       sim(V,V1,S3),S3>=95.

% it is possibly a merge of cora publication id X,Y, if they have similar titles T1,T2, and less similar author names AN1,AN2 and 
% have very similar volume V,V1 
active(X,Y) :- X!=Y,cora(ADD, AN, BT, D,ED,X, INS,J,M, N,PA,P,TEC,T,TY,V,YE),
      cora( ADD1, AN1, BT1, D1,ED1,Y, INS1,J1,M1,N1,PA1,P1,TEC1,T1,TY1,V1,YE1), 
       sim(T,T1,S1), S1>=90, sim(AN,AN1,S2), S2>=85,
       sim(V,V1,S3),S3>=95.

% it is possibly a merge of cora publication id X,Y, if they have similar titles T1,T2, and less similar journal names J1,J2 and 
% have less similar editors ED,ED1 
active(X,Y) :- X!=Y,cora(ADD, AN, BT, D,ED,X, INS,J,M, N,PA,P,TEC,T,TY,V,YE),
      cora( ADD1, AN1, BT1, D1,ED1,Y, INS1,J1,M1,N1,PA1,P1,TEC1,T1,TY1,V1,YE1), 
       sim(T,T1,S1), S1>=90, sim(J,J1,S2),S2>=85,
       sim(ED,ED1,S3),S3>=80.

% it is possibly a merge of cora publication id X,Y, if they have similar titles T1,T2, and less similar booktitle names BT1,BT2 and 
% have less similar editors ED,ED1 
active(X,Y) :- X!=Y,cora(ADD, AN, BT, D,ED,X, INS,J,M, N,PA,P,TEC,T,TY,V,YE),
      cora( ADD1, AN1, BT1, D1,ED1,Y, INS1,J1,M1,N1,PA1,P1,TEC1,T1,TY1,V1,YE1), 
           sim(T,T1,S1), S1>=90,sim(BT,BT1,S2),S2>=85,
       sim(ED,ED1,S3),S3>=80.

% it is possibly a merge of cora publication id X,Y, if they have similar titles T1,T2, and less similar author names AN1,AN2 and 
% have less similar editors ED,ED1 
active(X,Y) :- X!=Y,cora(ADD, AN, BT, D,ED,X, INS,J,M, N,PA,P,TEC,T,TY,V,YE),
      cora( ADD1, AN1, BT1, D1,ED1,Y, INS1,J1,M1,N1,PA1,P1,TEC1,T1,TY1,V1,YE1), 
       sim(T,T1,S1), S1>=90, sim(AN,AN1,S2), S2>=85,
       sim(ED,ED1,S3),S3>=80.

%----
% it is possibly a merge of cora publication id X,Y, if they have similar titles T1,T2, and less similar journal names J1,J2 and 
% have less similar publishers P,P1 
active(X,Y) :- X!=Y,cora(ADD, AN, BT, D,ED,X, INS,J,M, N,PA,P,TEC,T,TY,V,YE),
      cora( ADD1, AN1, BT1, D1,ED1,Y, INS1,J1,M1,N1,PA1,P1,TEC1,T1,TY1,V1,YE1), 
       sim(T,T1,S1), S1>=90, sim(J,J1,S2),S2>=85,
       sim(P,P1,S3),S3>=80.

% it is possibly a merge of cora publication id X,Y, if they have similar titles T1,T2, and less similar booktitle names BT1,BT2 and 
% have less similar publishers P,P1 
active(X,Y) :- X!=Y,cora(ADD, AN, BT, D,ED,X, INS,J,M, N,PA,P,TEC,T,TY,V,YE),
      cora( ADD1, AN1, BT1, D1,ED1,Y, INS1,J1,M1,N1,PA1,P1,TEC1,T1,TY1,V1,YE1), 
           sim(T,T1,S1), S1>=90,sim(BT,BT1,S2),S2>=85,
    sim(P,P1,S3),S3>=80.

% it is possibly a merge of cora publication id X,Y, if they have similar titles T1,T2, and less similar author names AN1,AN2 and 
% have less similar publishers P,P1 
active(X,Y) :- X!=Y,cora(ADD, AN, BT, D,ED,X, INS,J,M, N,PA,P,TEC,T,TY,V,YE),
      cora( ADD1, AN1, BT1, D1,ED1,Y, INS1,J1,M1,N1,PA1,P1,TEC1,T1,TY1,V1,YE1), 
       sim(T,T1,S1), S1>=90, sim(AN,AN1,S2), S2>=85,
  sim(P,P1,S3),S3>=80.



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% The following soft rules can be expressed all together with: 

%% active(X,Y) :- X!=Y,cora(ADD, AN, BT, D,ED,X, INS,J,M, N,PA,P,TEC,T,TY,V,YE),
%%    cora( ADD1, AN1, BT1, D1,ED1,Y, INS1,J1,M1,N1,PA1,P1,TEC,T1,TY1,V1,YE1), 
%%    sim(T,T1,S1), S1>=98, 
%%    #count{ 1:sim(J,J1,S2),S2>=80; 
%%    1:sim(BT,BT1,S2),S2>=80; 
%%    1:sim(AN,AN1,S2),S2>=85} = 1.

%% That is, it possibly is a merge of cora publication id X,Y, if they have extremely similar titles T1,T2, and 
%%  satisfy one of the conditions 
%%% {less similar journal name J,J1, 
%%% less similar booketitle BT,BT1, 
%%% less similar author names AN,AN1}.

% it is possibly a merge of cora publication id X,Y, if they have extremely similar titles T1,T2, and lesser similar journal names J1,J2
active(X,Y) :- X!=Y,cora(ADD, AN, BT, D,ED,X, INS,J,M, N,PA,P,TEC,T,TY,V,YE),
      cora( ADD1, AN1, BT1, D1,ED1,Y, INS1,J1,M1,N1,PA1,P1,TEC1,T1,TY1,V1,YE1), 
       sim(T,T1,S1), S1>=98, sim(J,J1,S2),S2>=80.

% it is possibly a merge of cora publication id X,Y, if they have extremely similar titles T1,T2, and lesser similar booktitle names BT1,BT2
active(X,Y) :- X!=Y,cora(ADD, AN, BT, D,ED,X, INS,J,M, N,PA,P,TEC,T,TY,V,YE),
      cora( ADD1, AN1, BT1, D1,ED1,Y, INS1,J1,M1,N1,PA1,P1,TEC1,T1,TY1,V1,YE1), 
       sim(T,T1,S1), S1>=98, sim(BT,BT1,S2),S2>=80.

% it is possibly a merge of cora publication id X,Y, if they have extremely similar titles T1,T2, and lesser similar author names AN1,AN2
active(X,Y) :- X!=Y,cora(ADD, AN, BT, D,ED,X, INS,J,M, N,PA,P,TEC,T,TY,V,YE),
      cora( ADD1, AN1, BT1, D1,ED1,Y, INS1,J1,M1,N1,PA1,P1,TEC1,T1,TY1,V1,YE1), 
       sim(T,T1,S1), S1>=98,   sim(AN,AN1,S2), S2>=80.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       



% it is possibly a merge of cora publication id X,Y, if they have identical titles T
active(X,Y) :- X!=Y,cora(ADD, AN, BT, D,ED,X, INS,J,M, N,PA,P,TEC,T,TY,V,YE),
      cora( ADD1, AN1, BT1, D1,ED1,Y, INS1,J1,M1,N1,PA1,P1,TEC1,T,TY1,V1,YE1),  not empty(T).     

 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    

% it is possibly a merge of cora publication id X,Y, if they have very similar titles T1,T2, and identical author name AN
active(X,Y) :- X!=Y,cora(ADD, AN, BT, D,ED,X, INS,J,M, N,PA,P,TEC,T,TY,V,YE),
      cora( ADD1, AN, BT1, D1,ED1,Y, INS1,J1,M1,N1,PA1,P1,TEC1,T1,TY1,V1,YE1),  sim(T,T1,S), S>=95, not empty(AN). 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    

%% The following soft rules can be expressed all together with: 

%% active(X,Y) :- X!=Y,cora(ADD, AN, BT, D,ED,X, INS,J,M, N,PA,P,TEC,T,TY,V,YE),
%%    cora( ADD1, AN1, BT1, D1,ED1,Y, INS1,J1,M1,N1,PA1,P1,TEC,T1,TY1,V1,YE1), 
%%    sim(T,T1,S1), S1>=95, 
%%    #count{ 1:J=J1; 
%%    1:BT=BT1} = 1, 
%%    #count{ 1:YE=YE1; 
%%    1:D=D1} = 1, 

%% That is, it possibly is a merge of cora publication id X,Y, if they have very similar titles T1,T2, and 
%%  satisfy one of the conditions 
%%% {have the same journal name J=J1, 
%%% have the same booketitle BT=BT1}
%% and satisfy one of the conditions 
%%% {published in the same year YE=YE1, 
%%% published in the same year D=D1 }.

% it is possibly a merge of cora publication id X,Y, if they have very similar titles T1,T2, and identical booktitle name BT and published in the same year YE
active(X,Y) :- X!=Y,cora(ADD, AN, BT, D,ED,X, INS,J,M, N,PA,P,TEC,T,TY,V,YE),
      cora( ADD1, AN1, BT, D1,ED1,Y, INS1,J1,M1,N1,PA1,P1,TEC1,T1,TY1,V1,YE),  sim(T,T1,S), S>=95, not empty(BT), not empty(YE). 

% it is possibly a merge of cora publication id X,Y, if they have very similar titles T1,T2, and identical journal name J and published in the same year YE
active(X,Y) :- X!=Y,cora(ADD, AN, BT, D,ED,X, INS,J,M, N,PA,P,TEC,T,TY,V,YE),
      cora( ADD1, AN1, BT1, D1,ED1,Y, INS1,J,M1,N1,PA1,P1,TEC1,T1,TY1,V1,YE),  sim(T,T1,S), S>=95, not empty(J), not empty(YE). 


% it is possibly a merge of cora publication id X,Y, if they have very similar titles T1,T2, and identical journal name J and published in the same date D
active(X,Y) :- X!=Y,cora(ADD, AN, BT, D,ED,X, INS,J,M, N,PA,P,TEC,T,TY,V,YE),
      cora( ADD1, AN1, BT, D,ED1,Y, INS1,J1,M1,N1,PA1,P1,TEC1,T1,TY1,V1,YE1),  sim(T,T1,S), S>=95, not empty(BT), not empty(D). 

% it is possibly a merge of cora publication id X,Y, if they have very similar titles T1,T2, and identical journal name J and published in the same date D
active(X,Y) :- X!=Y,cora(ADD, AN, BT, D,ED,X, INS,J,M, N,PA,P,TEC,T,TY,V,D),
      cora( ADD1, AN1, BT1, D,ED1,Y, INS1,J,M1,N1,PA1,P1,TEC1,T1,TY1,V1,YE1),  sim(T,T1,S), S>=95, not empty(J), not empty(D). 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% reject those publication merges where X is technical report TEC but Y is not
:- X!=Y, cora(ADD, AN, BT, D,ED,X, INS,J,M, N,PA,P,TEC,T,TY,V,YE),
     cora( ADD1, AN1, BT1, D1,ED1,Y, INS1,J1,M1,N1,PA1,P1,TEC1,T1,TY1,V1,YE1), eq(X,Y), 
      not empty(TEC), empty(TEC1).


{eq(X,Y)} :- active(X,Y).

empty(nan). empty("nan").empty("ーーー").

eq(X,Y) :- eq(X,Z),eq(Z,Y).
eq(X,Y) :- eq(Y,X).
