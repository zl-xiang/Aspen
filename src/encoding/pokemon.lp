


%%%  ability/3: ability can be obtained by pokemons
        %%% id 
        %%% generation: generation of the game 
        %%% is_main_series: whether in the main game series or not

%%% poke_ability/4: stores the mappings between pokemons and abilities
        %%% pokemon id
        %%% ability id
        %%% is_hidden: is hidden ability or not
        %%% slot: index of ability slot
% two abilities are a merge if they can be obtained by the same pokemon PID1,PID2 on the same ability slot SL
eq(X,Y) :- X!=Y, ability(X,GEN1,IMS1),ability(Y,GEN2,IMS2), 
           eq(X,X1), eq(Y,Y1),
           poke_ability(PID1,X1,IH1,SL), 
           poke_ability(PID2,Y1,IH2,SL), 
           eq(PID1,PID2).



%%%  item/5: items that occur in games
        %%% id 
        %%% category: the category of an item
        %%% cost: cost of gold to purchase
        %%% fling_power: fling damage durning battle
        %%% fling_effect: effect caused to opposing pokemon when flinging

%%% poke_item/4: stores the mappings between pokemons and items
        %%% pokemon id
        %%% version: version of the game
        %%% item id
        %%% rarity: how rare is the item be held by the wild pokemon

%%% item_name/3:
        %%% item_id
        %%% local_language
        %%% name: name of the item in the local language 
% two items are the same if they belong to the same category CT, and the same rarity R, and have similar names NA1,NA2 in the same local language LL, and can be held by the same pokemon PID1,PID2 
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


%%% species/18: stores species information in pokemon games, different pokemons can under the same species, but each pokemon belongs to only one species
        %%% id
        %%% generation
        %%% evolves_from_species: a reference of species id of its super species 
        %%% evolution_chain 
        %%% color
        %%% shape
        %%% habitat
        %%% gender_rate
        %%% capture_rate
        %%% base_happiness
        %%% is_baby
        %%% hatch_counter
        %%% has_gender_differences
        %%% growth_rate 
        %%% forms_switchable
        %%% is_legendary
        %%% is_mythical
        %%% conquest_order
 
 %%% species_name/4: 
        %%% species_id
        %%% local_language
        %%% name
        %%% genus
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


%%% move/14: moves that can be learned by pokemons and used in battles
        %%% id
        %%% generation
        %%% type
        %%% power
        %%% pp: how many times the move can be used durning battle at most
        %%% accuracy
        %%% priority
        %%% target
        %%% damage_class
        %%% effect
        %%% effect_chance
        %%% contest_type
        %%% contest_effect
        %%% super_contest_effect

%%% poke_move/6: stores the mappings between pokemons and moves
        %%% pokemon id
        %%% version: game version
        %%% move_id
        %%% pokemon_move_method: the method a move is taught to a pokemon
        %%% level: the minimal level required for a pokemon to learn the move
        %%% order: order of the move if its learned by leveling up
% two moves are the same if they are of the same type TP, have the same power PW and can be learnt by the same 
% pokemon PID1, PID2 at the same level LV, and in the same order O
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


%%% move_name/3:
        %%% move_id
        %%% local_language
        %%% name: name of the item in the local language 

% two moves are the same if they have very similar names NA1,NA2 in the same language LL, and are of the same type TP, and have the same PP, accuarcy ACC
%  power PW, priority PRI,  damage class DC,  effect EF
eq(X,Y):- move(X,GEN1,TP,PW,PP,ACC,PRI,TAG1,DC,EF,EFC1,CT1,CE1,SCE1), 
        move(Y,GEN2,TP,PW,PP,ACC,PRI,TAG2,DC,EF,EFC2,CT2,CE2,SCE2),
        eq(X,X1), eq(Y,Y1),
        move_name(X1,LL,NA1),
        move_name(Y1,LL,NA2),
        sim(NA1,NA2,S), S>=95, not empty(LL),not empty(TP), not empty(PP),not empty(ACC),not empty(DC),not empty(PRI).

% two species are the same if they have same pokemon PID1,PID2
eq(X,Y) :- species(X,GEN1,EFS1,EC1,CL1,SH1,HB1,GR1,CR1,BH1,IB1,HC1,HGD1,GRR1,FS1,IL1,IM1,CO1),
           species(Y,GEN2,EFS2,EC2,CL2,SH2,HB2,GR2,CR2,BH2,IB2,HC2,HGD2,GRR2,FS2,IL2,IM2,CO2),
           eq(X,X1), eq(Y,Y1),
           pokemon(PID1,N1,X1,HE1,WE1,BE1,IDF1),
           pokemon(PID2,N2,Y1,HE2,WE2,BE2,IDF2),
           eq(PID1,PID2).

 
% two pokemons are likely the same if they have very similar name N1,N2, and the same ability AID1,AID2, and the same is hidden IH on the same ability slot SL 
% is default form IDF
active(X,Y) :- pokemon(X,N1,SID1,HE,WE,BE,IDF),
           pokemon(Y,N2,SID2,HE1,WE1,BE1,IDF),
           sim(N1,N2,S),S>=95,
           not empty(IDF),
           eq(X,X1), eq(Y,Y1),
           poke_ability(X1,AID1,IH,SL), 
           poke_ability(Y1,AID2,IH,SL), 
           not empty(SL),
           eq(AID1,AID2).


%%% poke_type/3: All Pokémon creatures and their moves are assigned certain types from a set of type charts
       %%% pokemon_id,
       %%% type_id,
       %%% slot: type slot (at least 1, at most 2)
% two pokemons are likely the same if they have very similar names N1,N2, and belong to the same species SID1,SID2, and have the same 
% , and have the same type TP on the same type slot SL
active(X,Y) :- pokemon(X,N1,SID1,HE,WE,BE1,IDF),
           pokemon(Y,N2,SID2,HE2,WE2,BE2,IDF), 
           eq(SID1,SID2),
           sim(N1,N2,S), S>=95,
           not empty(IDF),
           eq(X,X1), eq(Y,Y1), not empty(TP), not empty(SL),
           poke_type(X1,TP,SL),poke_type(Y1,TP,SL).

% two pokemon are likely the same if they have similar names N1,N2, and have the same abilities AID1, AID2 on the same slot SL1, and have the same type TP on the same type slot SL
active(X,Y) :- pokemon(X,N1,SID1,HE,WE,BE1,IDF),
           pokemon(Y,N2,SID2,HE2,WE2,BE2,IDF), 
           sim(N1,N2,S), S>=90,
           eq(X,X2), eq(Y,Y2),
           eq(AID1,AID2),
           poke_ability(X2,AID1,IH1,SL1), 
           poke_ability(Y2,AID2,IH2,SL1),
           not empty(SL1), 
           not empty(IDF),
           eq(X,X1), eq(Y,Y1), not empty(TP), not empty(SL),
           poke_type(X1,TP,SL),poke_type(Y1,TP,SL).

% two abilities are likely the same if they have similar name NA1,NA2 in the same local language LL
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

% two moves are likey the same if they can be learned by the same pokemon PID1,PID2 at the same level LV
active(X,Y):- move(X,GEN1,TP1,PW1,PP1,ACC1,PRI1,TAG1,DC1,EF1,EFC1,CT1,CE1,SCE1), 
        move(Y,GEN2,TP2,PW2,PP2,ACC2,PRI2,TAG2,DC2,EF2,EFC2,CT2,CE2,SCE2), 
        eq(X,X1), eq(Y,Y1),
        poke_move(PID1,VER1,X1,PMM1,LV,O1),
        poke_move(PID2,VER2,Y1,PMM2,LV,O2),
        not empty(LV), % learnt at the same level
        eq(PID1,PID2). % by the same pokemon



% two items are likely the same if they have extremely similar name NA1,NA2 in the same local language LL
% and very similar description FT1,FT2 in the same language L
active(X,Y) :- X!=Y, item(X,CT,CO1,FP1,FE1),item(Y,CT,CO2,FP2,FE2),
        eq(X,X1), eq(Y,Y1),
        item_name(X1,LL,NA1),
        item_name(Y1,LL,NA2),
        sim(NA1,NA2,S), S>=98, not empty(LL),
        eq(X,X2), eq(Y,Y2),
        item_desc(X2,VER1,L,FT1),
        item_desc(Y2,VER2,L,FT2),
        not empty(L), sim(FT1,FT2,S1), S1>=95.


% two species are likely the same if they have very similar name NA1,NA2 in the same language LL and similar descriptions in the same language L
active(X,Y) :- X!=Y, species(X,GEN1,EFS1,EC1,CL1,SH1,HB1,GR1,CR1,BH1,IB1,HC1,HGD1,GRR1,FS1,IL1,IM1,CO1),
           species(Y,GEN2,EFS2,EC2,CL2,SH2,HB2,GR2,CR2,BH2,IB2,HC2,HGD2,GRR2,FS2,IL2,IM2,CO2),
           not empty(L),
           not empty(LL),
           eq(X,X1), eq(Y,Y1),
           spec_desc(X1,VER1,L,FT1), 
           spec_desc(Y1,VER2,L,FT2), sim(FT1,FT2,S), S>=90,
           eq(X,X2), eq(Y,Y2),
           spec_name(X2,LL,NA1,GN1),
           spec_name(Y2,LL,NA2,GN2), sim(NA1,NA2,S2), S2>=95.
                   
% two moves are likely the same if they have same base power PW and of the same type TP and have similar names NA1,NA2 in the same language LL
active(X,Y) :- X!=Y, move(X,GEN1,TP,PW,PP1,ACC1,PRI1,TAG1,DC,EF1,EFC1,CT1,CE1,SCE1), 
        move(Y,GEN2,TP,PW,PP2,AC2C,PRI2,TAG2,DC1,EF2,EFC2,CT2,CE2,SCE2),
        not empty(PW),
        not empty(TP),
        eq(X,X1), eq(Y,Y1), 
        move_name(X1,LL,NA1),
        move_name(Y1,LL,NA2),
        sim(NA1,NA2,S), S>=90, not empty(LL).


% reject pokemon merges X,Y that are of different species SPE1,SPE2
:- pokemon(X,N1,SPE1,HE1,WE1,BE1,IDF1),
   pokemon(Y,N2,SPE2,HE2,WE2,BE2,IDF2), eq(X,Y),            
   not eq(SPE1,SPE2).

% reject pokemon merges X,Y that are of different type TPID1,TPID2 on the same type slot S
:-  poke_type(X,TPID1,S), 
    poke_type(Y,TPID2,S),eq(X,Y), TPID1!=TPID2.          

% reject pokemon merges X,Y for any different values BST,BST2 on the same stat SID 
:- poke_stats(X,SID,BST,EF),eq(X,Y),
   poke_stats(Y,SID,BST2,EF2), 
   BST!=BST2.


{eq(X,Y)} :- active(X,Y).

empty(nan). empty("nan").empty("ーーー").

eq(X,Y) :- eq(X,Z),eq(Z,Y).
eq(X,Y) :- eq(Y,X).
