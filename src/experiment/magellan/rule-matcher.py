# Import py_entitymatching package
import py_entitymatching as em
import os
import pandas as pd
import time



def sim_rule_blocker(tbl1:tuple,tbl2:tuple,result_file='',rules:list=[]):
    # print(tbl1)
    # print(tbl2)
    A = em.read_csv_metadata(tbl1[0], key=tbl1[1])
    B = em.read_csv_metadata(tbl2[0], key=tbl2[1])
    
    block_f = em.get_features_for_blocking(A, B, validate_inferred_attr_types=False)
    # print(block_f)
    # check attribute correspondance
    # print(em._block_c['corres'])
    # em._atypes1['title'], em._atypes1['authors'], em._atypes1['venue'], em._atypes1['year']
    rb = em.RuleBasedBlocker()
    # Add rule : block tuples if name_name_lev(ltuple, rtuple) < 0.4
    for r in rules:
        #rb.add_rule(['title_title_jwn(ltuple, rtuple) < 0.4'], block_f)
        rb.add_rule(r, block_f)
    if '_name' in tbl1[0] :
        C = rb.block_tables(A, B, l_output_attrs=[tbl1[1],A.columns[1],A.columns[2]], r_output_attrs=[tbl2[1],B.columns[1],B.columns[2]], show_progress=False)
    elif '_desc' in tbl1[0]:
        C = rb.block_tables(A, B, l_output_attrs=[tbl1[1],A.columns[1],A.columns[3]], r_output_attrs=[tbl2[1],B.columns[1],B.columns[3]], show_progress=False)
    else:
         C = rb.block_tables(A, B, l_output_attrs=[tbl1[1]], r_output_attrs=[tbl2[1]], show_progress=False)
         
    #C.drop_duplicates(subset=C.columns.difference(['_id']))
    # print(C)
    C.to_csv(result_file, encoding='utf-8', index=False)
    # return result_file
    
    
def rule_matcher(l_tbl,r_tbl,candidates,rules:list, result_dir):
 
    path_A = l_tbl[0]
    path_B = r_tbl[0]
    key_A  = l_tbl[1]
    key_B = r_tbl[1]
    
    A = em.read_csv_metadata(path_A, key=key_A)
    B = em.read_csv_metadata(path_B, key=key_B)

    # Load the pre-labeled data
    S = em.read_csv_metadata(candidates, 
                            key='_id',
                            ltable=A, rtable=B, 
                            fk_ltable=f'ltable_{key_A}', fk_rtable=f'rtable_{key_B}')

    brm = em.BooleanRuleMatcher()
    # Generate a set of features
    F = em.get_features_for_matching(A, B, validate_inferred_attr_types=False)
    # print(F)
    for r in rules:
        brm.add_rule(r,F)

    # The first rule has two predicates, one comparing the titles and the other looking for an exact match of the years
    #brm.add_rule(r, F)
    # Rules can also be deleted from the rule-based matcherffiles
    #brm.delete_rule('_rule_1')
    brm.predict(S, target_attr='pred_label', append=True)
    print()
    S.to_csv(result_dir, encoding='utf-8', index=False)


def get_csv_filename(directory):
    # Convert the directory path to an absolute path if it's relative
    directory = os.path.abspath(directory)
    
    # Check if the directory exists
    if os.path.exists(directory):
        # If the directory is a file (ends with .csv), return its name
        if directory.endswith('.csv'):
            return os.path.basename(directory)
        else:
            # List all files in the directory
            files = os.listdir(directory)
            
            # Filter the list to include only CSV files
            csv_files = [file for file in files if file.endswith('.csv')]
            
            if csv_files:
                # Assuming there's only one CSV file in the directory, return its name
                return csv_files[0]
            else:
                print("No CSV files found in the directory '{}'.".format(directory))
                return None
    else:
        print("Directory or file '{}' does not exist.".format(directory))
        return None



class match_config:
    def __init__(self, name,tbl_pairs:list, b_rules:dict, m_rules:dict,) -> None:
        self.name = name
        if name in ['music']:
            self.base_dir = f'../dataset/{name}/50'
        elif name in [ 'pokemon']:
            self.base_dir = f'../dataset/{name}/50'
        elif name == 'music-corr':
            self.base_dir = f'../dataset/music/50-corr'
        else:
            self.base_dir = f'../dataset/{name}'
        self.tbl_pairs =tbl_pairs
        self.b_rules = b_rules
        self.m_rules = m_rules
        



def match(config:match_config):
    pairs = config.tbl_pairs
    time_sum = 0
    for i,p in enumerate(pairs):
        path_A = f'{config.base_dir}/{p[0][0]}.csv'
        path_B = f'{config.base_dir}/{p[1][0]}.csv'
        print(path_A)
        print(path_B)
        aname = get_csv_filename(path_A).replace('.csv','')
        bname = get_csv_filename(path_B).replace('.csv','')
        cand_dir = f'{config.base_dir}/{aname}-{bname}-cand.csv'
        sim_rule_blocker((path_A, p[0][1]),(path_B, p[1][1]),cand_dir,config.b_rules[i])
        match_dir = f'{config.base_dir}/{aname}-{bname}-match.csv'
        start = time.time()
        rule_matcher((path_A, p[0][1]),(path_B, p[1][1]),cand_dir,config.m_rules[i],match_dir)
        end = time.time()
        dur = end - start
        time_sum += dur
        print(dur)
    print(f'{config.name} spent {str(time_sum)}s for matching .')


def get_condition(attr_name, is_sim:bool, threshold=None, is_long:bool=False)->str:
    sim = '_jwn(ltuple, rtuple) >= '
    sim_long = '_jac_qgm_3_qgm_3(ltuple, rtuple) >= '
    join = '_exm(ltuple, rtuple) == 1'
    if is_sim:
        if is_long: return f'{attr_name}_{attr_name}{sim_long}{str(threshold)}'
        else: return f'{attr_name}_{attr_name}{sim}{str(threshold)}'
    else:
        return f'{attr_name}_{attr_name}{join}'
        
    
    
def dblp_match_config():
    pairs = [(('dblp','id'),('acm','id'))]
    b_rules = [
        [['title_title_jac_qgm_3_qgm_3(ltuple, rtuple) < 0.4']]
    ]
    m_rules = [
        [
        ['title_title_jwn(ltuple, rtuple) >= 0.95', 'year_year_exm(ltuple, rtuple) == 1','venue_venue_exm(ltuple, rtuple) == 1'],
        ['title_title_jwn(ltuple, rtuple) >= 0.95', 'authors_authors_jwn(ltuple, rtuple) >= 0.95','year_year_exm(ltuple, rtuple) == 1'],
        ['title_title_jwn(ltuple, rtuple) >= 0.90', 'authors_authors_jwn(ltuple, rtuple) >= 0.85','year_year_exm(ltuple, rtuple) == 1'],
        ['title_title_jwn(ltuple, rtuple) >= 0.95'],
        ['title_title_jwn(ltuple, rtuple) >= 0.90','year_year_exm(ltuple, rtuple) == 1'],
        ['title_title_jwn(ltuple, rtuple) >= 0.90', 'authors_authors_jwn(ltuple, rtuple) >= 0.90'],]
        
        
    ]
    
    return  match_config('dblp',tbl_pairs=pairs,b_rules=b_rules,m_rules=m_rules)


def cora_match_config():
    title_98 = get_condition(attr_name='title',is_sim=True,threshold=0.98)
    
    title_95 = get_condition(attr_name='title',is_sim=True,threshold=0.95)
    authors_95 =  get_condition(attr_name='authors',is_sim=True,threshold=0.95)
    editor_95 =  get_condition(attr_name='editor',is_sim=True,threshold=0.95)
    journal_95 =  get_condition(attr_name='journal',is_sim=True,threshold=0.95)
    booktitle_95 = get_condition(attr_name='booktitle',is_sim=True,threshold=0.95)
    publisher_95 = get_condition(attr_name='publisher',is_sim=True,threshold=0.95)
    
    year_95 = get_condition(attr_name='year',is_sim=True,threshold=0.95)
    date_95 = get_condition(attr_name='date',is_sim=True,threshold=0.95)
    volume_95 = get_condition(attr_name='volume',is_sim=True,threshold=0.95)
    pages_95 = get_condition(attr_name='pages',is_sim=True,threshold=0.95)
    
    title_90 = get_condition(attr_name='title',is_sim=True,threshold=0.90)
    authors_90 =  get_condition(attr_name='authors',is_sim=True,threshold=0.90)
    editor_90 =  get_condition(attr_name='editor',is_sim=True,threshold=0.90)
    journal_90 =  get_condition(attr_name='journal',is_sim=True,threshold=0.90)
    booktitle_90 = get_condition(attr_name='booktitle',is_sim=True,threshold=0.90)
    publisher_90 = get_condition(attr_name='publisher',is_sim=True,threshold=0.90)
    tech_90 = get_condition(attr_name='tech',is_sim=True,threshold=0.90)
    address_90 = get_condition(attr_name='address',is_sim=True,threshold=0.90)
    
    
    title_85 = get_condition(attr_name='title',is_sim=True,threshold=0.85)
    authors_85 =  get_condition(attr_name='authors',is_sim=True,threshold=0.85)
    editor_85 =  get_condition(attr_name='editor',is_sim=True,threshold=0.85)
    journal_85 =  get_condition(attr_name='journal',is_sim=True,threshold=0.85)
    booktitle_85 = get_condition(attr_name='booktitle',is_sim=True,threshold=0.85)
    publisher_85 = get_condition(attr_name='publisher',is_sim=True,threshold=0.85)
    
    title_80 = get_condition(attr_name='title',is_sim=True,threshold=0.80)
    authors_80=  get_condition(attr_name='authors',is_sim=True,threshold=0.80)
    editor_80=  get_condition(attr_name='editor',is_sim=True,threshold=0.80)
    journal_80 =  get_condition(attr_name='journal',is_sim=True,threshold=0.80)
    booktitle_80 = get_condition(attr_name='booktitle',is_sim=True,threshold=0.80)
    publisher_80 = get_condition(attr_name='publisher',is_sim=True,threshold=0.80)
    
    
    same_year = get_condition(attr_name='year',is_sim=False)
    same_date = get_condition(attr_name='date',is_sim=False)
    same_tech =  get_condition(attr_name='tech',is_sim=False)
    same_address =  get_condition(attr_name='address',is_sim=False)
    same_pages=  get_condition(attr_name='pages',is_sim=False)
    same_volume =  get_condition(attr_name='volume',is_sim=False)
    same_title = get_condition(attr_name='title',is_sim=False)
    same_authors = get_condition(attr_name='authors',is_sim=False)
    same_journal = get_condition(attr_name='journal',is_sim=False)
    same_booktitle = get_condition(attr_name='booktitle',is_sim=False)
    
    
    pairs = [(('cora','id'),('cora','id'))]
    b_rules = [
        ['title_title_jac_qgm_3_qgm_3(ltuple, rtuple) < 0.4']
    ]
    
    rule_ones = []
    rule_one_fix = [title_95,same_tech]
    rule_one_var1 = [journal_95,booktitle_95,authors_95]
    rule_one_var2 = [same_address,same_date,same_pages,same_year,same_volume,editor_95,publisher_95]
    for var1 in rule_one_var1:
        fix = rule_one_fix.copy()
        fix.append(var1)
        for var2 in rule_one_var2:
            _fix = fix.copy()
            _fix.append(var2)
            rule_ones.append(_fix)
            
    rule_two = [[title_90,tech_90]]
    rule_threes = []
    rule_three_fix = [title_90,year_95]
    rule_three_var1 = [journal_85,booktitle_85,authors_85]
    for var1 in rule_three_var1:
        fix = rule_three_fix.copy()
        fix.append(var1)
        rule_threes.append(fix)
    
        
    rule_fours = []
    rule_4_fix = [title_85]
    rule_4_var1 = [journal_95,booktitle_95,authors_90]
    rule_4_var2 = [year_95,date_95]
    for var1 in rule_4_var1:
        fix = rule_4_fix.copy()
        fix.append(var1)
        for var2 in rule_4_var2:
            _fix = fix.copy()
            _fix.append(var2)
            rule_fours.append(_fix)
            
    rule_fives = []
    rule_5_fix = [title_90]
    rule_5_var1 = [journal_85,booktitle_85,authors_80]
    rule_5_var2 = [address_90,pages_95,volume_95,editor_80,publisher_80]
    for var1 in rule_5_var1:
        fix = rule_5_fix.copy()
        fix.append(var1)
        for var2 in rule_5_var2:
            _fix = fix.copy()
            _fix.append(var2)
            rule_fives.append(_fix)
    
    
    rule_sixes = []
    rule_6_fix = [title_98]
    rule_6_var1 = [journal_80,booktitle_80,authors_85]
    #rule_5_var2 = [address_90,pages_95,volume_95,editor_80,publisher_80]
    for var1 in rule_6_var1:
        fix = rule_6_fix.copy()
        fix.append(var1)
        rule_sixes.append(fix)
        
    rule_7 = [[same_title]]
    rule_8 = [[title_95,same_authors]]
    
    
    rule_9s = []
    rule_9_fix = [title_95]
    rule_9_var1=  [same_journal, same_booktitle]
    rule_9_var2 = [same_date,same_year]
    for var1 in rule_9_var1:
        fix = rule_9_fix.copy()
        fix.append(var1)
        for var2 in rule_9_var2:
            _fix = fix.copy()
            _fix.append(var2)
            rule_9s.append(_fix)
            
    rule_10s = [[title_85,booktitle_95,date_95], [title_90,journal_85,date_95],[title_90,booktitle_85,date_95],[title_90,authors_85,date_95]]

    m_rules = [rule_ones + rule_two + rule_threes + rule_fours + rule_fives + rule_sixes + rule_7 + rule_8 + rule_9s + rule_10s]
    
    return  match_config('cora',tbl_pairs=pairs,b_rules=b_rules,m_rules=m_rules)
    
    
def imdb_match_config():
    pairs = [(('title_basics','tconst'),('title_basics','tconst')), (('name_basics','nconst'),('name_basics','nconst'))]
    b_rules = [
       [ ['originalTitle_originalTitle_jac_qgm_3_qgm_3(ltuple, rtuple) < 0.4','primaryTitle_primaryTitle_jac_qgm_3_qgm_3(ltuple, rtuple) < 0.4']],
        [['primaryName_primaryName_jac_qgm_3_qgm_3(ltuple, rtuple) < 0.4']] 
        
    ]
    pt_98 = get_condition('primaryTitle',is_sim=True,threshold=0.98)
    ot_98 = get_condition('originalTitle',is_sim=True,threshold=0.98)
    
    pt_95 = get_condition('primaryTitle',is_sim=True,threshold=0.95)
    pt_90 = get_condition('primaryTitle',is_sim=True,threshold=0.90)
    
    pm_98 = get_condition('primaryName',is_sim=True,threshold=0.98)
    pm_90 = get_condition('primaryName',is_sim=True,threshold=0.90)
    
    same_gen = get_condition('genres',is_sim=False)
    same_start_year = get_condition('startYear',is_sim=False)
    same_is_adult = get_condition('isAdult',is_sim=False)
    same_title_type = get_condition('titleType',is_sim=False)
    same_pp = get_condition('primaryProfession',is_sim=False)
    
    m_rules = [
       [ [pt_98, ot_98,same_gen,same_start_year,same_title_type,same_is_adult],
        [pt_90, same_start_year,same_gen],
        [pt_98, same_title_type,same_is_adult]],
       [[pm_98],[pm_90,same_pp]],
    ]
    
    return  match_config('imdb',tbl_pairs=pairs,b_rules=b_rules,m_rules=m_rules)    

def music_match_config():
    rels = ['area','artist','artist_credit','label','medium','place','recording','release','release_group','track']
    pairs = []
    for r in rels:
        pairs.append(((r,r),(r,r)))
    print(pairs)
    
    b_rules = [
        [['name_name_jac_qgm_3_qgm_3(ltuple, rtuple) < 0.4']],
         [['name_name_jac_qgm_3_qgm_3(ltuple, rtuple) < 0.4']],
        [['name_name_jac_qgm_3_qgm_3(ltuple, rtuple) < 0.4']],
        [['name_name_jac_qgm_3_qgm_3(ltuple, rtuple) < 0.4']],
        [['release_release_exm(ltuple, rtuple) != 1']],
        [['name_name_jac_qgm_3_qgm_3(ltuple, rtuple) < 0.4']],
        [['name_name_jac_qgm_3_qgm_3(ltuple, rtuple) < 0.4']],
        [['name_name_jac_qgm_3_qgm_3(ltuple, rtuple) < 0.4']],
         [['name_name_jac_qgm_3_qgm_3(ltuple, rtuple) < 0.4']],
         [['name_name_jac_qgm_3_qgm_3(ltuple, rtuple) < 0.4']],
        
    ]
    
    
    name_90 = get_condition('name',is_sim=True,threshold=0.90)
    name_95 = get_condition('name',is_sim=True,threshold=0.95)
    name_85 = get_condition('name',is_sim=True,threshold=0.85)
    
    sort_name_90 =  get_condition('sort_name',is_sim=True,threshold=0.95)
    
    # area
    same_area_type = get_condition('area_type',is_sim=False)
    same_end = get_condition('ended',is_sim=False)
    # artist
    same_area = get_condition('area',is_sim=False)
    same_artist_type = get_condition('artist_type',is_sim=False)
    same_begin_date_year = get_condition('begin_date_year',is_sim=False)
    same_gender = get_condition('gender',is_sim=False)
    # artist_credit
    same_artist_count = get_condition('artist_count',is_sim=False)
    same_ref_count = get_condition('ref_count',is_sim=False)
    # label
    same_label_type = get_condition('label_type',is_sim=False)
    # medium
    same_release = get_condition('release',is_sim=False)
    same_position = get_condition('position',is_sim=False)
    # place
    same_place_type = get_condition('place_type',is_sim=False)
    same_coordinates = get_condition('coordinates',is_sim=False)
    # recording
    same_artist_credit = get_condition('artist_credit',is_sim=False)
    same_length = get_condition('length',is_sim=False)
    same_video = get_condition('video',is_sim=False)
    #release
    same_barcode = get_condition('barcode',is_sim=False)
    same_release_group = get_condition('release_group',is_sim=False)
    same_status = get_condition('status',is_sim=False)
    #release_group
    same_type = get_condition('type',is_sim=False)
    #track
    same_medium = get_condition('medium',is_sim=False)
    same_number =get_condition('number',is_sim=False)
    same_is_data_track = get_condition('is_data_track',is_sim=False)
    
    
    m_rules = [
        # area
       [ [name_90,same_area_type,same_end], ],
       # artist
       [[name_95,same_area], [name_90,same_artist_type,same_begin_date_year ],[name_90,sort_name_90,same_gender]],
       #artist_credit
       [[name_90,same_artist_count,same_ref_count],],
       # label
       [[name_90,same_area,same_label_type],[name_90,same_label_type],[name_95]],
       # medium
       [[same_release,same_position],],
       # place
       [[name_90,same_place_type,same_area],[same_coordinates],[name_90,same_area]],
       # recording
       [[same_artist_credit,name_90,same_length,same_video],[name_85,same_length,same_video], ],
       # release
       [[same_barcode],[same_release_group,name_90,same_status],[name_90,same_artist_credit]],
       # release_group
       [[name_90,same_artist_credit],[name_90,same_type]],
       # track
       [[name_90,same_medium,same_position],[name_90,same_artist_credit],[name_85,same_length,same_number,same_is_data_track]]
    ]
    
    return  match_config('music',tbl_pairs=pairs,b_rules=b_rules,m_rules=m_rules)  



def music_match_config():
    rels = ['area','artist','artist_credit','label','medium','place','recording','release','release_group','track']
    pairs = []
    for r in rels:
        pairs.append(((r,r),(r,r)))
    print(pairs)
    
    b_rules = [
        [['name_name_jac_qgm_3_qgm_3(ltuple, rtuple) < 0.4']],
         [['name_name_jac_qgm_3_qgm_3(ltuple, rtuple) < 0.4']],
        [['name_name_jac_qgm_3_qgm_3(ltuple, rtuple) < 0.4']],
        [['name_name_jac_qgm_3_qgm_3(ltuple, rtuple) < 0.4']],
        [['release_release_exm(ltuple, rtuple) != 1']],
        [['name_name_jac_qgm_3_qgm_3(ltuple, rtuple) < 0.4']],
        [['name_name_jac_qgm_3_qgm_3(ltuple, rtuple) < 0.4']],
        [['name_name_jac_qgm_3_qgm_3(ltuple, rtuple) < 0.4']],
         [['name_name_jac_qgm_3_qgm_3(ltuple, rtuple) < 0.4']],
         [['name_name_jac_qgm_3_qgm_3(ltuple, rtuple) < 0.4']],
        
    ]
    
    
    name_90 = get_condition('name',is_sim=True,threshold=0.90)
    name_95 = get_condition('name',is_sim=True,threshold=0.95)
    name_85 = get_condition('name',is_sim=True,threshold=0.85)
    
    sort_name_90 =  get_condition('sort_name',is_sim=True,threshold=0.95)
    
    # area
    same_area_type = get_condition('area_type',is_sim=False)
    same_end = get_condition('ended',is_sim=False)
    # artist
    same_area = get_condition('area',is_sim=False)
    same_artist_type = get_condition('artist_type',is_sim=False)
    same_begin_date_year = get_condition('begin_date_year',is_sim=False)
    same_gender = get_condition('gender',is_sim=False)
    # artist_credit
    same_artist_count = get_condition('artist_count',is_sim=False)
    same_ref_count = get_condition('ref_count',is_sim=False)
    # label
    same_label_type = get_condition('label_type',is_sim=False)
    # medium
    same_release = get_condition('release',is_sim=False)
    same_position = get_condition('position',is_sim=False)
    # place
    same_place_type = get_condition('place_type',is_sim=False)
    same_coordinates = get_condition('coordinates',is_sim=False)
    # recording
    same_artist_credit = get_condition('artist_credit',is_sim=False)
    same_length = get_condition('length',is_sim=False)
    same_video = get_condition('video',is_sim=False)
    #release
    same_barcode = get_condition('barcode',is_sim=False)
    same_release_group = get_condition('release_group',is_sim=False)
    same_status = get_condition('status',is_sim=False)
    #release_group
    same_type = get_condition('type',is_sim=False)
    #track
    same_medium = get_condition('medium',is_sim=False)
    same_number =get_condition('number',is_sim=False)
    same_is_data_track = get_condition('is_data_track',is_sim=False)
    
    
    m_rules = [
        # area
       [ [name_90,same_area_type,same_end], ],
       # artist
       [[name_95,same_area], [name_90,same_artist_type,same_begin_date_year ],[name_90,sort_name_90,same_gender]],
       #artist_credit
       [[name_90,same_artist_count,same_ref_count],],
       # label
       [[name_90,same_area,same_label_type],[name_90,same_label_type],[name_95]],
       # medium
       [[same_release,same_position],],
       # place
       [[name_90,same_place_type,same_area],[same_coordinates],[name_90,same_area]],
       # recording
       [[same_artist_credit,name_90,same_length,same_video],[name_85,same_length,same_video], ],
       # release
       [[same_barcode],[same_release_group,name_90,same_status],[name_90,same_artist_credit]],
       # release_group
       [[name_90,same_artist_credit],[name_90,same_type]],
       # track
       [[name_90,same_medium,same_position],[name_90,same_artist_credit],[name_85,same_length,same_number,same_is_data_track]]
    ]
    
    return  match_config('music',tbl_pairs=pairs,b_rules=b_rules,m_rules=m_rules) 

def music_corr_match_config():
    rels = ['area','artist','artist_credit','label','medium','place','recording','release','release_group','track']
    pairs = []
    for r in rels:
        pairs.append(((r,r),(r,r)))
    # print(pairs)
    
    b_rules = [
        [['name_name_jac_qgm_3_qgm_3(ltuple, rtuple) < 0.4']],
         [['name_name_jac_qgm_3_qgm_3(ltuple, rtuple) < 0.4']],
        [['name_name_jac_qgm_3_qgm_3(ltuple, rtuple) < 0.4']],
        [['name_name_jac_qgm_3_qgm_3(ltuple, rtuple) < 0.4']],
        [['release_release_exm(ltuple, rtuple) != 1']],
        [['name_name_jac_qgm_3_qgm_3(ltuple, rtuple) < 0.4']],
        [['name_name_jac_qgm_3_qgm_3(ltuple, rtuple) < 0.4']],
        [['name_name_jac_qgm_3_qgm_3(ltuple, rtuple) < 0.4']],
         [['name_name_jac_qgm_3_qgm_3(ltuple, rtuple) < 0.4']],
         [['name_name_jac_qgm_3_qgm_3(ltuple, rtuple) < 0.4']],
        
    ]
    
    
    name_90 = get_condition('name',is_sim=True,threshold=0.90)
    name_95 = get_condition('name',is_sim=True,threshold=0.95)
    name_85 = get_condition('name',is_sim=True,threshold=0.85)
    
    sort_name_90 =  get_condition('sort_name',is_sim=True,threshold=0.95)
    
    # area
    same_area_type = get_condition('area_type',is_sim=False)
    same_end = get_condition('ended',is_sim=False)
    # artist
    same_area = get_condition('area',is_sim=False)
    same_artist_type = get_condition('artist_type',is_sim=False)
    same_begin_date_year = get_condition('begin_date_year',is_sim=False)
    same_gender = get_condition('gender',is_sim=False)
    # artist_credit
    same_artist_count = get_condition('artist_count',is_sim=False)
    same_ref_count = get_condition('ref_count',is_sim=False)
    # label
    same_label_type = get_condition('label_type',is_sim=False)
    # medium
    same_release = get_condition('release',is_sim=False)
    same_position = get_condition('position',is_sim=False)
    # place
    same_place_type = get_condition('place_type',is_sim=False)
    same_coordinates = get_condition('coordinates',is_sim=False)
    address_90 = get_condition('address',is_sim=True,threshold=0.90)
    # recording
    same_artist_credit = get_condition('artist_credit',is_sim=False)
    same_length = get_condition('length',is_sim=False)
    same_video = get_condition('video',is_sim=False)
    #release
    same_barcode = get_condition('barcode',is_sim=False)
    same_release_group = get_condition('release_group',is_sim=False)
    same_status = get_condition('status',is_sim=False)
    #release_group
    same_type = get_condition('type',is_sim=False)
    #track
    same_medium = get_condition('medium',is_sim=False)
    same_number =get_condition('number',is_sim=False)
    same_is_data_track = get_condition('is_data_track',is_sim=False)
    
    
    m_rules = [
        # area
       [ [name_90,same_area_type,same_end], ],
       # artist
       [[name_95,same_area], [name_90,same_artist_type,same_begin_date_year ],[name_90,sort_name_90,same_gender]],
       #artist_credit
       [[name_90,same_artist_count,same_ref_count],],
       # label
       [[name_90,same_area,same_label_type],[name_90,same_label_type],[name_90,same_end],[name_90,same_area],],
       # medium
       [[same_release,same_position],],
       # place
       [[name_90,same_place_type,same_area],[same_coordinates],[name_90,same_area],[name_90,same_place_type,address_90],],
       # recording
       [[same_artist_credit,name_90,same_length,same_video],[name_85,same_length,same_video], [name_85,same_artist_credit,same_video],],
       # release
       [[same_barcode],[same_release_group,name_90,same_status],[name_90,same_artist_credit],[same_release_group,same_artist_credit]],
       # release_group
       [[name_90,same_artist_credit],[name_90,same_type]],
       # track
       [[name_95,same_medium,same_position],[name_95,same_artist_credit],[same_medium,same_position,same_artist_credit],[name_95,same_length],[name_85,same_length,same_number,same_is_data_track]]
    ]
    
    return  match_config('music-corr',tbl_pairs=pairs,b_rules=b_rules,m_rules=m_rules)  



def music_match_config():
    rels = ['area','artist','artist_credit','label','medium','place','recording','release','release_group','track']
    pairs = []
    for r in rels:
        pairs.append(((r,r),(r,r)))
    print(pairs)
    
    b_rules = [
        [['name_name_jac_qgm_3_qgm_3(ltuple, rtuple) < 0.4']],
         [['name_name_jac_qgm_3_qgm_3(ltuple, rtuple) < 0.4']],
        [['name_name_jac_qgm_3_qgm_3(ltuple, rtuple) < 0.4']],
        [['name_name_jac_qgm_3_qgm_3(ltuple, rtuple) < 0.4']],
        [['release_release_exm(ltuple, rtuple) != 1']],
        [['name_name_jac_qgm_3_qgm_3(ltuple, rtuple) < 0.4']],
        [['name_name_jac_qgm_3_qgm_3(ltuple, rtuple) < 0.4']],
        [['name_name_jac_qgm_3_qgm_3(ltuple, rtuple) < 0.4']],
         [['name_name_jac_qgm_3_qgm_3(ltuple, rtuple) < 0.4']],
         [['name_name_jac_qgm_3_qgm_3(ltuple, rtuple) < 0.4']],
        
    ]
    
    
    name_90 = get_condition('name',is_sim=True,threshold=0.90)
    name_97 = get_condition('name',is_sim=True,threshold=0.97)
    name_95 = get_condition('name',is_sim=True,threshold=0.95)
    name_85 = get_condition('name',is_sim=True,threshold=0.85)
    
    same_release = get_condition('release',is_sim=False)
    same_position = get_condition('position',is_sim=False)
    # place
    same_place_type = get_condition('place_type',is_sim=False)
    same_coordinates = get_condition('coordinates',is_sim=False)
    # recording
    same_artist_credit = get_condition('artist_credit',is_sim=False)
    same_length = get_condition('length',is_sim=False)
    sort_name_90 =  get_condition('sort_name',is_sim=True,threshold=0.95)
    sort_name_97 =  get_condition('sort_name',is_sim=True,threshold=0.97)
    # area
    same_area_type = get_condition('area_type',is_sim=False)
    same_end = get_condition('ended',is_sim=False)
    # artist
    same_area = get_condition('area',is_sim=False)
    same_artist_type = get_condition('artist_type',is_sim=False)
    same_begin_date_year = get_condition('begin_date_year',is_sim=False)
    same_gender = get_condition('gender',is_sim=False)
    # artist_credit
    same_artist_count = get_condition('artist_count',is_sim=False)
    same_ref_count = get_condition('ref_count',is_sim=False)
    # label
    same_label_type = get_condition('label_type',is_sim=False)
    # medium
    same_release = get_condition('release',is_sim=False)
    same_position = get_condition('position',is_sim=False)
    # place
    same_place_type = get_condition('place_type',is_sim=False)
    same_coordinates = get_condition('coordinates',is_sim=False)
    # recording
    same_artist_credit = get_condition('artist_credit',is_sim=False)
    same_length = get_condition('length',is_sim=False)
    same_video = get_condition('video',is_sim=False)
    #release
    same_barcode = get_condition('barcode',is_sim=False)
    same_release_group = get_condition('release_group',is_sim=False)
    same_status = get_condition('status',is_sim=False)
    #release_group
    same_type = get_condition('type',is_sim=False)
    #track
    same_medium = get_condition('medium',is_sim=False)
    same_number =get_condition('number',is_sim=False)
    same_is_data_track = get_condition('is_data_track',is_sim=False)
    
    
    m_rules = [
        # area
       [ [name_90,same_area_type,same_end], ],
       # artist
       [[name_95,same_area], [name_90,same_artist_type,same_begin_date_year ],[name_97,sort_name_97]],
       #artist_credit
       [[name_90,same_artist_count,same_ref_count],],
       # label
       [[name_90,same_area,same_label_type],[name_90,same_label_type],[name_95]],
       # medium
       [[same_release,same_position],],
       # place
       [[name_90,same_place_type,same_area],[same_coordinates],[name_90,same_area]],
       # recording
       [[same_artist_credit,name_90,same_length,same_video],[name_85,same_length,same_video], ],
       # release
       [[same_barcode],[same_release_group,name_90,same_status],[name_90,same_artist_credit]],
       # release_group
       [[name_90,same_artist_credit],[name_90,same_type]],
       # track
       [[name_95,same_medium,same_position],[name_95,same_artist_credit],[name_85,same_length,same_number,same_is_data_track]]
    ]
    
    return  match_config('music',tbl_pairs=pairs,b_rules=b_rules,m_rules=m_rules) 



def pokemon_match_config():
    rels = ['ability_name','ability_desc','item_name','item_desc','move_name','move_desc','species','species_name','species_desc','pokemon']
    pairs = []
    for r in rels:
        if r.endswith('_name') or r.endswith('_desc'):
            real_r = r.replace('_name','') if r.endswith('_name') else  r.replace('_desc','')
            pairs.append(((r,'id'),(r,'id')))
        else:
            pairs.append(((r,r),(r,r)))
    #print(pairs)
    
    b_rules = [
        [['name_name_jac_qgm_3_qgm_3(ltuple, rtuple) < 0.4','local_language_local_language_exm(ltuple,rtuple) == 1']],
         [['flavor_text_flavor_text_jac_qgm_3_qgm_3(ltuple, rtuple) < 0.4','language_language_exm(ltuple,rtuple) == 1','version_group_version_group_exm(ltuple,rtuple) == 1' ]],
        [['name_name_jac_qgm_3_qgm_3(ltuple, rtuple) < 0.4','local_language_local_language_exm(ltuple,rtuple) == 1']],
         [['flavor_text_flavor_text_jac_qgm_3_qgm_3(ltuple, rtuple) < 0.4','language_language_exm(ltuple,rtuple) == 1','version_group_version_group_exm(ltuple,rtuple) == 1' ]],
        [['name_name_jac_qgm_3_qgm_3(ltuple, rtuple) < 0.4','local_language_local_language_exm(ltuple,rtuple) == 1']],
         [['is_legendary_is_legendary_species_exm(ltuple,rtuple) != 1','growth_rate_growth_rate_exm(ltuple,rtuple) != 1']],
        [['name_name_jac_qgm_3_qgm_3(ltuple, rtuple) < 0.4','local_language_local_language_exm(ltuple,rtuple) == 1']],
         [['flavor_text_flavor_text_jac_qgm_3_qgm_3(ltuple, rtuple) < 0.4','language_language_exm(ltuple,rtuple) == 1','version_group_version_group_exm(ltuple,rtuple) == 1' ]],
         [['name_name_jac_qgm_3_qgm_3(ltuple, rtuple) < 0.4']],
        
    ]
    
    
    name_90 = get_condition('name',is_sim=True,threshold=0.90)
    name_97 = get_condition('name',is_sim=True,threshold=0.97)
    name_98 = get_condition('name',is_sim=True,threshold=0.98)
    name_95 = get_condition('name',is_sim=True,threshold=0.95)
    name_85 = get_condition('name',is_sim=True,threshold=0.85)
    
    flavor_text_90 = get_condition('flavor_text',is_sim=True,threshold=0.90)
    flavor_text_95 = get_condition('flavor_text',is_sim=True,threshold=0.95)
    
    
    same_local_language = get_condition('local_language',is_sim=False)
    same_language = get_condition('language',is_sim=False)
    same_evolves_from_species = get_condition('evolves_from_species',is_sim=False)
    same_generation = get_condition('generation',is_sim=False)
    same_shape = get_condition('shape',is_sim=False)
    same_habitat =get_condition('habitat',is_sim=False)
    same_gender_rate = get_condition('gender_rate',is_sim=False)
    same_capture_rate = get_condition('capture_rate',is_sim=False)
    
    

    
    m_rules = [
       # ability name
       [ [name_90,same_local_language]],
       # ability desc
       [[flavor_text_90,same_language]],
       # item name
       [[name_98,same_local_language],],
       # item desc
       [[flavor_text_95,same_language]],
       # move_name
       [[name_90,same_local_language],],
       # move desc
       [[name_90,same_language]],
       # species
       [[same_evolves_from_species,same_generation,same_shape,same_habitat,same_gender_rate,same_capture_rate]],
       # species_name
       [ [name_95,same_local_language]],       
        # species_desc
       [[name_90,same_local_language],]
    ]
    
    return  match_config('pokemon',tbl_pairs=pairs,b_rules=b_rules,m_rules=m_rules) 


def pokemon_match_config2():
    rels = ['ability','item','move','species','pokemon']
    pairs = []
    for r in rels:
        if r.endswith('_name') or r.endswith('_desc'):
            real_r = r.replace('_name','') if r.endswith('_name') else  r.replace('_desc','')
            pairs.append(((r,'id'),(r,'id')))
        else:
            pairs.append(((r,r),(r,r)))
    #print(pairs)
    
    b_rules = [
        [['is_main_series_is_main_series_exm(ltuple,rtuple) != 1']],
         [['category_category_exm(ltuple, rtuple) != 1']],
        [['type_type_exm(ltuple, rtuple) != 1','power_power_exm(ltuple,rtuple) != 1']],
         [['evolves_from_species_evolves_from_species_exm(ltuple, rtuple) != 1']],
        [['name_name_jac_qgm_3_qgm_3(ltuple, rtuple) < 0.4']],
        
    ]
    
    
    name_90 = get_condition('name',is_sim=True,threshold=0.90)
    name_97 = get_condition('name',is_sim=True,threshold=0.97)
    name_98 = get_condition('name',is_sim=True,threshold=0.98)
    name_95 = get_condition('name',is_sim=True,threshold=0.95)
    name_85 = get_condition('name',is_sim=True,threshold=0.85)
    
    flavor_text_90 = get_condition('flavor_text',is_sim=True,threshold=0.90)
    flavor_text_95 = get_condition('flavor_text',is_sim=True,threshold=0.95)
    
    same_is_main_series = get_condition('is_main_series',is_sim=False)
    same_category = get_condition('category',is_sim=False)
    same_cost = get_condition('cost',is_sim=False)
    same_fling_power = get_condition('fling_power',is_sim=False)
    
    
    same_type = get_condition('type',is_sim=False)
    same_power = get_condition('power',is_sim=False)
    accuracy = get_condition('accuracy',is_sim=False)
    effect = get_condition('effect',is_sim=False)
    damage_class = get_condition('damage_class',is_sim=False)
    priority = get_condition('priority',is_sim=False)
    
    
    species = get_condition('species',is_sim=False)
    
    same_local_language = get_condition('local_language',is_sim=False)
    same_language = get_condition('language',is_sim=False)
    same_evolves_from_species = get_condition('evolves_from_species',is_sim=False)
    same_generation = get_condition('generation',is_sim=False)
    same_shape = get_condition('shape',is_sim=False)
    same_habitat =get_condition('habitat',is_sim=False)
    same_gender_rate = get_condition('gender_rate',is_sim=False)
    same_capture_rate = get_condition('capture_rate',is_sim=False)
    
    

    
    m_rules = [
       # ability 
       [ [same_generation,same_is_main_series]],
       # item name
       [[same_category,same_cost,same_fling_power]],
       # move_name
       [[same_type,same_power,accuracy,effect,damage_class,priority],],
       # species
       [[same_evolves_from_species,same_generation,same_shape,same_habitat,same_gender_rate,same_capture_rate]],
       # pokemon
       [ [name_90,species]],       
    ]
    
    return  match_config('pokemon',tbl_pairs=pairs,b_rules=b_rules,m_rules=m_rules) 


def process_csv_files(directory):
    # Iterate through files in the directory
    for filename in os.listdir(directory):
        if filename.endswith("_desc.csv") or filename.endswith("_name.csv"):
            # Construct the full file path
            file_path = os.path.join(directory, filename)
            
            # Read CSV file into a pandas DataFrame
            df = pd.read_csv(file_path)
            
            # Insert extra column 'id'
            df.insert(0, 'id', range(1, len(df) + 1))
            
            # Save DataFrame to CSV with original filename
            new_filename = filename.split('.')[0]  # Remove extension
            new_file_path = os.path.join(directory, new_filename + '.csv')
            df.to_csv(new_file_path, index=False)
            print(f"Processed: {file_path} -> {new_file_path}")



if __name__ == "__main__":
   #process_csv_files('../dataset/pokemon/py-em')
    #df = pd.read_csv('../dataset/cora/cora.tsv', sep='\t')
    match(pokemon_match_config2())

