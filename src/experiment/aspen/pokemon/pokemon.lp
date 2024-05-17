
%eq_pm_stats(PID11,SID,BST,EF) :- poke_stats(T,PID1,SID,BST,EF), pokemon(X,PID1,IDT1,SPE1,HE1,WE1,BE1,O1,IDF1), 
 %                   eq(X,XA), pokemon(XA,PID11,IDT11,SPE11,HE11,WE11,BE11,O11,IDF11), poke_stats(T2,PID11,SID,BST,EF).

%not_same_stats(X,Y) :-  poke_stats(T,PID11,SID,BST,EF), pokemon(X,PID1,IDT1,SPE1,HE1,WE1,BE1,O1,IDF1), eq(X,XA),
 %                       pokemon(XA,PID11,IDT11,SPE11,HE11,WE11,BE11,O11,IDF11), 
  %                      poke_stats(T2,PID22,SID,BST2,EF2), pokemon(Y,PID2,IDT2,SPE2,HE2,WE2,BE2,O2,IDF2), eq(Y,YA),
   %                     pokemon(YA,PID22,IDT22,SPE22,HE22,WE22,BE22,O22,IDF22), BST!=BST2.

%not_same_type(X,Y) :-  poke_type(TID1,PID11,TPID1,S), pokemon(X,PID1,IDT1,SPE1,HE1,WE1,BE1,O1,IDF1), eq(X,XA),
 %                       pokemon(XA,PID11,IDT11,SPE11,HE11,WE11,BE11,O11,IDF11), 
  %                     poke_type(TID2,PID22,TPID2,S2), pokemon(Y,PID2,IDT2,SPE2,HE2,WE2,BE2,O2,IDF2), eq(Y,YA),
   %                     pokemon(YA,PID22,IDT22,SPE22,HE22,WE22,BE22,O22,IDF22), TPID1!=TPID2.                

% pair of pokemon is a merge if they have exactly the same (set of) stats and of the same (set of) types
%eq(X,Y) :- pokemon(X,PID1,IDT1,SPE1,HE1,WE1,BE,O1,IDF),
 %          pokemon(Y,PID2,IDT2,SPE2,HE2,WE2,BE,O2,IDF),
  %         sim(IDT1,IDT2,S),S>=95, 
   %        not empty(BE),
    %       not not_same_stats(X,Y),
     %      not not_same_type(X,Y), not empty(IDF).


% two abilities are a merge if they can be obtained by the same pokemon PID1,PID2 on the same ability slot SL 
eq(X,Y) :- X!=Y, ability(X,GEN1,IMS1),ability(Y,GEN2,IMS2), 
           eq(X,X1), eq(Y,Y1),
           poke_ability(PID1,X1,IH1,SL), 
           poke_ability(PID2,Y1,IH2,SL), 
           eq(PID1,PID2).

% low coverage
% two items are the same if they belong to the same category CT, and have similar names NA1,NA2 in the same local language LL, and can be held by the same pokemon PID1,PID2 
% and with the same rarity R
eq(X,Y) :- X!=Y, item(X,CT,CO1,FP1,FE1),item(Y,CT,CO2,FP2,FE2),not empty(CT),
                eq(X,X1), eq(Y,Y1),
                item_name(X1,LL,NA1),
                item_name(Y1,LL,NA2),
                not empty(LL),
                sim(NA1,NA2,S2),S2>=98,
                eq(X,X2), eq(Y,Y2),
               poke_item(PID1,VER1,X2,R),
               poke_item(PID2,VER2,Y2,R),
               not empty(R),
               eq(PID1,PID2).

% acurate but really low coverage
% two species are the same if the have the same super species EFS1,EFS2 and have the same gender rate GR
% and the same shape SH and the same habitat and belong to the same generation GEN with the same capture rate CR
% and have similar names NA1, NA2 in the same local language LL
eq(X,Y) :- species(X,GEN,EFS1,EC1,CL1,SH,HB,GR,CR,BH1,IB1,HC1,HGD1,GRR1,FS1,IL1,IM1,CO1),
           species(Y,GEN,EFS2,EC2,CL2,SH,HB,GR,CR,BH2,IB2,HC2,HGD2,GRR2,FS2,IL2,IM2,CO2),
           eq(EFS1,EFS2),
           eq(X,X1), eq(Y,Y1),
           spec_name(X1,LL,NA1,GN1),
           spec_name(Y1,LL,NA2,GN2), sim(NA1,NA2,S2), S2>=95, not empty(LL),
           not empty(GEN), not empty(SH), not empty(HB), not empty(GR), not empty(CR).



% acurate by low coverage
% two moves are the same if they are of the same type TP, have the same power PW and can be learnt by the same 
% pokemon PID1, PID2, and have similar name NA1, NA2 in the same local language LL
eq(X,Y):- move(X,GEN1,TP,PW,PP1,ACC1,PRI1,TAG1,DC1,EF,EFC1,CT1,CE1,SCE1), 
        move(Y,GEN2,TP,PW,PP2,ACC2,PRI2,TAG2,DC2,EF,EFC2,CT2,CE2,SCE2), 
        not empty(TP), % added of the same type
        eq(X,X1), eq(Y,Y1),
        poke_move(PID1,VER1,X1,PMM1,LV,O),
        poke_move(PID2,VER2,Y1,PMM2,LV,O),
        not empty(PW), % same base power
        not empty(TP), % same type
        not empty(LV), % learnt at the same level
        not empty(O), % of same order
        eq(PID1,PID2). % by the same pokemon

% acurate but low coverage
% two moves are the same if they are of the same type TP, and have the same PP, accuarcy ACC
%  power PW, priority PRI,  damage class DC,  effect EF
eq(X,Y):- move(X,GEN1,TP,PW,PP,ACC,PRI,TAG1,DC,EF,EFC1,CT1,CE1,SCE1), 
        move(Y,GEN2,TP,PW,PP,ACC,PRI,TAG2,DC,EF,EFC2,CT2,CE2,SCE2),
        eq(X,X1), eq(Y,Y1),
        move_name(X1,LL,NA1),
        %eq(Y,YA), ability(YA,AID22,IDT22,GEN22,IMS22), 
        move_name(Y1,LL,NA2),
        sim(NA1,NA2,S), S>=95, not empty(LL),not empty(TP), not empty(PP),not empty(ACC),not empty(DC),not empty(PRI).

% two species are the same if they have same pokemon PID1,PID2
eq(X,Y) :- species(X,GEN1,EFS1,EC1,CL1,SH1,HB1,GR1,CR1,BH1,IB1,HC1,HGD1,GRR1,FS1,IL1,IM1,CO1),
           species(Y,GEN2,EFS2,EC2,CL2,SH2,HB2,GR2,CR2,BH2,IB2,HC2,HGD2,GRR2,FS2,IL2,IM2,CO2),
           eq(X,X1), eq(Y,Y1),
           pokemon(PID1,N1,X1,HE1,WE1,BE1,IDF1),
           pokemon(PID2,N2,Y1,HE2,WE2,BE2,IDF2),
           eq(PID1,PID2).


 % not accurate    
% two pokemons are the same if they have the same ability AID1,AID2, and the same is hidden IH on the same ability slot SL 
% and belong to the same species and is default form IDF
active(X,Y) :- pokemon(X,N1,SID1,HE,WE,BE,IDF),
           pokemon(Y,N2,SID2,HE1,WE1,BE1,IDF),
           sim(N1,N2,S),S>=95,
           not empty(IDF),
           eq(X,X1), eq(Y,Y1),
           poke_ability(X1,AID1,IH,SL), 
           poke_ability(Y1,AID2,IH,SL), 
           not empty(SL),
           eq(AID1,AID2).


% not accurate and low coverage
% two pokemons are likely the same if the belong to the same species SID1,SID2, and have the same 
% height HE, same weight WE, and have the same type TP on the same type slot SL
active(X,Y) :- pokemon(X,N1,SID1,HE,WE,BE1,IDF),
           pokemon(Y,N2,SID2,HE2,WE2,BE2,IDF), 
           eq(SID1,SID2),
           %not empty(HE),
           %not empty(WE), 
           sim(N1,N2,S), S>=95,
           not empty(IDF),
           eq(X,X1), eq(Y,Y1), not empty(TP), not empty(SL),
           poke_type(X1,TP,SL),poke_type(Y1,TP,SL).
           %not empty(IDF). % added

% increases the grounding time significantly (without eq(SID1,SID2))
% not really useful
active(X,Y) :- pokemon(X,N1,SID1,HE,WE,BE1,IDF),
           pokemon(Y,N2,SID2,HE2,WE2,BE2,IDF), 
           % eq(SID1,SID2),
           sim(N1,N2,S), S>=90,
           %not empty(HE),
           %not empty(WE),
           eq(X,X2), eq(Y,Y2),
           eq(AID1,AID2),
           poke_ability(X2,AID1,IH1,SL1), 
           poke_ability(Y2,AID2,IH2,SL1),
           not empty(SL1), 
           not empty(IDF),
           eq(X,X1), eq(Y,Y1), not empty(TP), not empty(SL),
           poke_type(X1,TP,SL),poke_type(Y1,TP,SL).


% increases coverage, but precision dropped
% two abilities are the same if they have similar name NA1,NA2 in the same local language LL
% and similar description FT1,FT2 in the same language L
active(X,Y) :- X!=Y, ability(X,GEN,IMS1),ability(Y,GEN2,IMS2),
        eq(X,X1), eq(Y,Y1),
        ability_name(X1,LL,NA1),
        ability_name(Y1,LL,NA2),
        sim(NA1,NA2,S), S>=90, not empty(LL),
        eq(X,X2), eq(Y,Y2),
        ability_desc(X2,VER1,L,FT1),
        ability_desc(Y2,VER2,L,FT2),
        not empty(L), sim(FT1,FT2,S1), S1>=90.

active(X,Y):- move(X,GEN1,TP1,PW1,PP1,ACC1,PRI1,TAG1,DC1,EF1,EFC1,CT1,CE1,SCE1), 
        move(Y,GEN2,TP2,PW2,PP2,ACC2,PRI2,TAG2,DC2,EF2,EFC2,CT2,CE2,SCE2), 
        eq(X,X1), eq(Y,Y1),
        poke_move(PID1,VER1,X1,PMM1,LV,O),
        poke_move(PID2,VER2,Y1,PMM2,LV,O),
        not empty(LV), % learnt at the same level
        not empty(O), % of same order
        eq(PID1,PID2). % by the same pokemon


%%%% soft rules


% boost coverage, but accuracy dropped by (almost) half
% two items are the same if the have the same category CT, and similar names NA1, NA2 in the same local language
% and similar descriptions FT1, FT2 in the same language L
%active(X,Y) :- X!=Y, item(X,CT,CO1,FP1,FE1),item(Y,CT,CO2,FP2,FE2),not empty(CT),
 %       eq(X,X1), eq(Y,Y1),
  %      item_name(X1,LL,NA1),
   %     item_name(Y1,LL,NA2),
    %    sim(NA1,NA2,S), S>=95, not empty(LL),
     %   eq(X,X2), eq(Y,Y2),
      %  item_desc(X2,VER1,L,FT1),
       % item_desc(Y2,VER2,L,FT2),
        %not empty(L), sim(FT1,FT2,S1), S1>=95.

active(X,Y) :- X!=Y, item(X,CT,CO1,FP1,FE1),item(Y,CT,CO2,FP2,FE2),
        eq(X,X1), eq(Y,Y1),
        item_name(X1,LL,NA1),
        item_name(Y1,LL,NA2),
        sim(NA1,NA2,S), S>=98, not empty(LL),
        eq(X,X2), eq(Y,Y2),
        item_desc(X2,VER1,L,FT1),
        item_desc(Y2,VER2,L,FT2),
        not empty(L), sim(FT1,FT2,S1), S1>=95.


% this is expensive, 300s grounding time increase
% but really accurate and decent coverage
% two species are the same if they have similar name NA1,NA2 in the same language LL and similar descriptions in the same language L
active(X,Y) :- X!=Y, species(X,GEN1,EFS1,EC1,CL1,SH1,HB1,GR1,CR1,BH1,IB1,HC1,HGD1,GRR1,FS1,IL1,IM1,CO1),
           species(Y,GEN2,EFS2,EC2,CL2,SH2,HB2,GR2,CR2,BH2,IB2,HC2,HGD2,GRR2,FS2,IL2,IM2,CO2),
           % not empty(GEN),
           not empty(L),
           not empty(LL),
           eq(X,X1), eq(Y,Y1),
           spec_desc(X1,VER1,L,FT1), 
           spec_desc(Y1,VER2,L,FT2), sim(FT1,FT2,S), S>=90,
           eq(X,X2), eq(Y,Y2),
           spec_name(X2,LL,NA1,GN1),
           spec_name(Y2,LL,NA2,GN2), sim(NA1,NA2,S2), S2>=95.
           
% not useful, and decrease the precision          
% two moves are likely the same if they have similar name NA1,NA2 in the same language LL and can be learnt by the same pokemon 
% similar descriptions in the same language L
active(X,Y) :- X!=Y, move(X,GEN1,TP,PW,PP1,ACC1,PRI1,TAG1,DC,EF1,EFC1,CT1,CE1,SCE1), 
        move(Y,GEN2,TP,PW,PP2,AC2C,PRI2,TAG2,DC1,EF2,EFC2,CT2,CE2,SCE2),
        not empty(PW),
        % not empty(PP),
        % not empty(PRI),
        %not empty(ACC),
        not empty(TP),
        eq(X,X1), eq(Y,Y1),
        %eq(X,XA), ability(XA,AID11,IDT11,GEN11,IMS11), 
        move_name(X1,LL,NA1),
        %eq(Y,YA), ability(YA,AID22,IDT22,GEN22,IMS22), 
        move_name(Y1,LL,NA2),
        %eq(X,X1), eq(Y,Y1),
        %eq(X,X2), eq(Y,Y2),
        %poke_move(PID1,VER3,X2,PMM1,LV1,O1),
        %poke_move(PID2,VER4,Y2,PMM2,LV2,O2),
        %eq(PID1,PID2),
        sim(NA1,NA2,S), S>=90, not empty(LL).




%:- pokemon(X,PID1,IDT3,SPE1,HE1,WE1,BE1,O1,IDF1),
  % pokemon(Y,PID2,IDT4,SPE2,HE2,WE2,BE2,O2,IDF2), eq(X,Y), not_same_type(X,Y).

%:- pokemon(X,PID1,IDT3,SPE1,HE1,WE1,BE1,O1,IDF1),
   %pokemon(Y,PID2,IDT4,SPE2,HE2,WE2,BE2,O2,IDF2), eq(X,Y), not_same_stats(X,Y).


   

:- pokemon(X,N1,SPE1,HE1,WE1,BE1,IDF1),
   pokemon(Y,N2,SPE2,HE2,WE2,BE2,IDF2), eq(X,Y),            
   not eq(SPE1,SPE2).

%:- eq(X,Y),not eq(AID1,AID2),poke_ability(X,AID1,IH1,SL), poke_ability(Y,AID2,IH2,SL).


:-  poke_type(X,TPID1,S), 
    poke_type(Y,TPID2,S),eq(X,Y), TPID1!=TPID2.          

:- poke_stats(X,SID,BST,EF),eq(X,Y),
   poke_stats(Y,SID,BST2,EF2), 
   BST!=BST2.       


% pokemon and moves
% pokemon and items

% same species evolve from different species

% same move have different base power and different types


