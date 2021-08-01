# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from google.cloud import storage
from pandas.io import gbq
import pandas as pd
import pickle
import re
import os

class PatentLandscapeExpander:
    """Class for L1&L2 expansion as 'Automated Patent Landscaping' describes.
    This object takes a seed set and a Google Cloud BigQuery project name and
    exposes methods for doing expansion of the project. The logical entry-point
    to the class is load_from_disk_or_do_expansion, which checks for cached
    expansions for the given self.seed_name, and if a previous run is available
    it will load it from disk and return it; otherwise, it does L1 and L2
    expansions, persists it in a cached 'data/[self.seed_name]/' directory,
    and returns the data to the caller.
    """
    seed_file = None
    # BigQuery must be enabled for this project
    bq_project = None
    google_dataset = 'patents-public-data:patents.publications_latest'
    utility_dataset = 'landscaping-318416:patents.utility_patents'
    #tmp_table = 'patents._tmp'
    l1_tmp_table = 'patents._l1_tmp'
    l2_tmp_table = 'patents._l2_tmp'
    antiseed_tmp_table = 'patents._antiseed_tmp'
    country_codes = set(['US'])
    num_anti_seed_patents = 15000

    # ratios and multipler for finding uniquely common CPC codes from seed set
    min_ratio_of_code_to_seed = None
    min_seed_multiplier = None

    # persisted expansion information
    training_data_full_df = None
    seed_df = None
    seed_patents_df = None
    l1_patents_df = None
    l2_patents_df = None
    anti_seed_df = None
    seed_data_path = None

    def __init__(self, seed_file, seed_name, bq_project=None, google_dataset=None, utility_dataset=None, num_antiseed=None, min_ratio_of_code_to_seed = None, min_seed_multiplier = None):
        self.seed_file = seed_file
        self.seed_data_path = os.path.join('data', seed_name)

        if bq_project is not None:
            self.bq_project = bq_project
        if google_dataset is not None:
            self.google_dataset = google_dataset
        if utility_dataset is not None:
            self.utility_dataset = utility_dataset
        if num_antiseed is not None:
            self.num_anti_seed_patents = num_antiseed
        if min_ratio_of_code_to_seed is not None:
            self.min_ratio_of_code_to_seed = min_ratio_of_code_to_seed
        if min_seed_multiplier is not None:
            self.min_seed_multiplier = min_seed_multiplier
    
    def load_seed_pubs(self, seed_file=None):
        if seed_file is None:
            seed_file = self.seed_file

        seed_df = pd.read_csv(seed_file, header=None, names=['publication_number', 'ExpansionLevel'], dtype={'publication_number': 'str', 'ExpansionLevel': 'str'})

        return seed_df
        
    def load_backward_citations_from_pubs(self, pub_series):
        tmp_table = 'patents._tmp_citations'
        self.load_df_to_bq_tmp(pd.DataFrame(pub_series, columns=['publication_number']), tmp_table=tmp_table)

        backward_citations_query = '''
            SELECT
              p.publication_number,
              STRING_AGG(DISTINCT citations.publication_number) AS backward_citations
            FROM
              `patents-public-data.patents.publications` p,
              UNNEST(citation) AS citations,
              `{}` as tmp
            WHERE
              p.publication_number = tmp.publication_number
            GROUP BY p.publication_number
            ;
        '''.format(tmp_table)

        print('Loading backward citations from provided publication numbers.')
        backward_citations_df = gbq.read_gbq(
            query=backward_citations_query,
            project_id=self.bq_project,
            dialect='standard')

        return backward_citations_df
    
    def load_forward_citations_from_pubs(self, pub_series):
        tmp_table = 'patents._tmp_citations'
        self.load_df_to_bq_tmp(pd.DataFrame(pub_series, columns=['publication_number']), tmp_table=tmp_table)

        forward_citations_query = '''
            SELECT
              tmp.publication_number,
              STRING_AGG(DISTINCT p.publication_number) AS forward_citations
            FROM
              `patents-public-data.patents.publications` p,
              UNNEST(citation) AS citations,
              `{}` as tmp             
            WHERE
              citations.publication_number = tmp.publication_number
            GROUP BY tmp.publication_number
            ;
        '''.format(tmp_table)

        print('Loading forward citations from provided publication numbers.')
        forward_citations_df = gbq.read_gbq(
            query=forward_citations_query,
            project_id=self.bq_project,
            dialect='standard')

        return forward_citations_df

    def load_family_members_from_pubs(self, pub_series):
        tmp_table = 'patents._tmp_family'
        self.load_df_to_bq_tmp(pd.DataFrame(pub_series, columns=['publication_number']), tmp_table=tmp_table)

        family_members_query = '''
        SELECT 
          DISTINCT a.publication_number,
        FROM
          `patents-public-data.patents.publications` a
        RIGHT JOIN (
          SELECT
            family_id
          FROM
            `patents-public-data.patents.publications` AS p,
            `{}` AS tmp
          WHERE
            p.publication_number = tmp.publication_number) AS b
        ON
          a.family_id = b.family_id
        WHERE country_code = "US"
            ;
        '''.format(tmp_table)

        print('Loading family ids from provided publication numbers.')
        family_members_df = gbq.read_gbq(
            query=family_members_query,
            project_id=self.bq_project,
            dialect='standard')

        return family_members_df

    def bq_get_num_total_patents(self):
        num_patents_query = """
            SELECT
              COUNT(publication_number) AS num_patents
            FROM
              `patents-public-data.patents.publications`
            WHERE
              country_code = 'US' AND kind_code IN ('A', 'A1', 'A2', 'A9', 'B1', 'B2')
        """
        num_patents_df = gbq.read_gbq(
            query=num_patents_query,
            project_id=self.bq_project,
            dialect='standard')
        return num_patents_df

    def get_cpc_counts(self, seed_publications=None):
        where_clause = '1=1'
        if seed_publications is not None:
            where_clause = """
            b.publication_number IN
                (
                {}
                )
            """.format(",".join("'" + seed_publications + "'"))

        cpc_counts_query = """
            SELECT
              cpcs.code,
              COUNT(cpcs.code) AS cpc_count
            FROM
              `patents-public-data.patents.publications` AS b,
              UNNEST(cpc) AS cpcs
            WHERE
            {}
            AND cpcs.code != ''
            AND country_code = 'US'
            AND kind_code IN ('A', 'A1', 'A2', 'A9', 'B1', 'B2')
            GROUP BY cpcs.code
            ORDER BY cpc_count DESC;
            """.format(where_clause)

        return gbq.read_gbq(
            query=cpc_counts_query,
            project_id=self.bq_project,
            dialect='standard')

    def compute_uniquely_common_cpc_codes_for_seed(self, seed_df):
        '''
        Queries for CPC counts across all US patents and all Seed patents, then finds the CPC codes
        that are 50x more common in the Seed set than the rest of the patent corpus (and also appear in
        at least 5% of Seed patents). This then returns a Pandas dataframe of uniquely common codes
        as well as the table of CPC counts for reference. Note that this function makes several
        BigQuery queries on multi-terabyte datasets, so expect it to take a couple minutes.
        
        You should call this method like:
        uniquely_common_cpc_codes, cpc_counts_df = \
            expander.compute_uniquely_common_cpc_codes_for_seed(seed_df)
            
        where seed_df is the result of calling load_seed_pubs() in this class.
        '''

        print('Querying for all US CPC Counts')
        us_cpc_counts_df = self.get_cpc_counts()
        print('Querying for Seed Set CPC Counts')
        seed_cpc_counts_df = self.get_cpc_counts(seed_df.publication_number)
        print("Querying to find total number of US patents")
        num_patents_df = self.bq_get_num_total_patents()
        num_seed_patents = seed_df.count().values[0]
        num_us_patents = num_patents_df['num_patents'].values[0]
        print('Total number of US utility patents: {}'.format(num_us_patents))

        # Merge/join the dataframes on CPC code, suffixing them as appropriate
        cpc_counts_df = us_cpc_counts_df.merge(
            seed_cpc_counts_df, on='code', suffixes=('_us', '_seed')) \
            .sort_values(ascending=False, by=['cpc_count_seed'])

        # For each CPC code, calculate the ratio of how often the code appears
        #  in the seed set vs the number of total seed patents
        cpc_counts_df['cpc_count_to_num_seeds_ratio'] = cpc_counts_df.cpc_count_seed / num_seed_patents
        # Similarly, calculate the ratio of CPC document frequencies vs total number of US patents
        cpc_counts_df['cpc_count_to_num_us_ratio'] = cpc_counts_df.cpc_count_us / num_us_patents
        # Calculate how much more frequently a CPC code occurs in the seed set vs full corpus of US patents
        cpc_counts_df['seed_relative_freq_ratio'] = \
            cpc_counts_df.cpc_count_to_num_seeds_ratio / cpc_counts_df.cpc_count_to_num_us_ratio

        # We only care about codes that occur at least ~5% of the time in the seed set
        # AND are 50x more common in the seed set than the full corpus of US patents
        uniquely_common_cpc_codes = cpc_counts_df[
            (cpc_counts_df.cpc_count_to_num_seeds_ratio >= self.min_ratio_of_code_to_seed)
            &
            (cpc_counts_df.seed_relative_freq_ratio >= self.min_seed_multiplier)]

        return uniquely_common_cpc_codes, cpc_counts_df


    def get_set_of_refs_filtered_by_country(self, seed_refs_series, country_codes):
        '''
        Uses the refs column of the BigQuery on the seed set to compute the set of
        unique references out of the Seed set.
        '''

        all_relevant_refs = set()
        for refs in seed_refs_series:
            for ref in refs.split(','):
                country_code = re.sub(r'(\w+)-(\w+)-\w+', r'\1', ref)
                if country_code in country_codes:
                    all_relevant_refs.add(ref)

        return all_relevant_refs

    # Expansion Functions
    def load_df_to_bq_tmp(self, df, tmp_table):
        '''
        This function inserts the provided dataframe into a temp table in BigQuery, which
        is used in other parts of this class (e.g. L1 and L2 expansions) to join on by
        patent number.
        '''
        print('Loading dataframe with cols {}, shape {}, to {}'.format(
            df.columns, df.shape, tmp_table))
        gbq.to_gbq(
            dataframe=df,
            destination_table=tmp_table,
            project_id=self.bq_project,
            if_exists='replace')

        print('Completed loading temp table.')

    def expand_l2(self, refs_series):
        self.load_df_to_bq_tmp(pd.DataFrame(refs_series, columns=['pub_num']), self.l2_tmp_table)

        expansion_query = '''
            SELECT
              b.publication_number,
              'L2' AS ExpansionLevel
            FROM
              `patents-public-data.patents.publications` AS b,
              `{}` as tmp
            WHERE b.publication_number = tmp.pub_num
            ;
        '''.format(self.l2_tmp_table)

        #print(expansion_query)
        expansion_df = gbq.read_gbq(
            query=expansion_query,
            project_id=self.bq_project,
            dialect='standard')

        return expansion_df

    def expand_l1(self, cpc_codes_series, refs_series):
        self.load_df_to_bq_tmp(pd.DataFrame(refs_series, columns=['pub_num']), self.l1_tmp_table)

        cpc_where_clause = ",".join("'" + cpc_codes_series + "'")

        expansion_query = '''
            SELECT DISTINCT publication_number, ExpansionLevel
            FROM
            (
            SELECT
              b.publication_number,
              'L1' as ExpansionLevel
            FROM
              `patents-public-data.patents.publications` AS b,
              UNNEST(cpc) AS cpcs
            WHERE
            (
                cpcs.code IN
                (
                {}
                )
            )
            AND country_code IN ('US')

            UNION ALL

            SELECT
              b.publication_number,
              'L1' as ExpansionLevel
            FROM
              `patents-public-data.patents.publications` AS b,
              `{}` as tmp
            WHERE
            (
                b.publication_number = tmp.pub_num
            )
            )
            ;
        '''.format(cpc_where_clause, self.l1_tmp_table)

        #print(expansion_query)
        expansion_df = gbq.read_gbq(
            query=expansion_query,
            project_id=self.bq_project,
            dialect='standard')

        return expansion_df

    def anti_seed(self, seed_expansion_series):
        self.load_df_to_bq_tmp(pd.DataFrame(seed_expansion_series, columns=['pub_num']), self.antiseed_tmp_table)

        anti_seed_query = '''
            SELECT DISTINCT
              b.publication_number,
              'AntiSeed' AS ExpansionLevel,
              rand() as random_num
            FROM
              `landscaping-318416.patents.utility_patents` AS b
            LEFT OUTER JOIN `{}` AS tmp ON b.publication_number = tmp.pub_num
            WHERE tmp.pub_num IS NULL
            ORDER BY random_num
            LIMIT {}
            # TODO: randomize results
            ;
        '''.format(self.antiseed_tmp_table, self.num_anti_seed_patents)

        #print('Anti-seed query:\n{}'.format(anti_seed_query))
        anti_seed_df = gbq.read_gbq(
            query=anti_seed_query,
            project_id=self.bq_project,
            dialect='standard')

        return anti_seed_df

    def load_training_data_from_pubs(self, training_publications_df):
        tmp_table = 'patents._tmp_training'
        self.load_df_to_bq_tmp(df=training_publications_df, tmp_table=tmp_table)

        training_data_query = '''
            SELECT DISTINCT
              p.publication_number,
              title.text as title_text,
              abstract.text as abstract_text,
              STRING_AGG(DISTINCT citations.publication_number) AS refs,
              STRING_AGG(DISTINCT cpcs.code) AS cpcs
            FROM
              `patents-public-data.patents.publications` p,
              `{}` as tmp,
              UNNEST(p.title_localized) AS title,
              UNNEST(p.abstract_localized) AS abstract,
              UNNEST(p.citation) AS citations,
              UNNEST(p.cpc) AS cpcs
            WHERE
              p.publication_number = tmp.publication_number
              AND citations.publication_number != ''
              AND cpcs.code != ''
            GROUP BY p.publication_number, title.text, abstract.text
            ;
        '''.format(tmp_table)

        print('Loading patent texts from provided publication numbers.')
        #print('Training data query:\n{}'.format(training_data_query))
        training_data_df = gbq.read_gbq(
            query=training_data_query,
            project_id=self.bq_project,
            dialect='standard',
            configuration = {'query': {'useQueryCache': True, 'allowLargeResults': True}})

        return training_data_df

    def do_full_expansion(self):
        '''
        Does a full expansion on seed set as described in paper, using seed set
        to derive an anti-seed for use in supervised learning stage.
        
        Call this method like:
        seed_patents_df, l1_patents_df, l2_patents_df, anti_seed_patents = \
            expander.do_full_expansion(seed_file)
        '''
        seed_df = self.load_seed_pubs(self.seed_file)

        seed_patents_df = self.load_seeds_from_bq(seed_df)

        # Level 1 Expansion
        ## getting unique seed CPC codes
        uniquely_common_cpc_codes, cpc_counts_df = \
            self.compute_uniquely_common_cpc_codes_for_seed(seed_df)
        ## getting all the references out of the seed set
        all_relevant_refs = self.get_set_of_refs_filtered_by_country(
            seed_patents_df.refs, self.country_codes)
        print('Got {} relevant seed refs'.format(len(all_relevant_refs)))
        ## actually doing expansion with CPC and references
        l1_patents_df = self.expand_l1(
            uniquely_common_cpc_codes.code, pd.Series(list(all_relevant_refs)))
        print('Shape of L1 expansion: {}'.format(l1_patents_df.shape))
        l1_refs_df = self.expand_l1_refs(l1_patents_df[['publication_number']])

        # Level 2 Expansion
        l2_refs = self.get_set_of_refs_filtered_by_country(
            l1_refs_df.refs, self.country_codes)
        print('Got {} relevant L1->L2 refs'.format(len(l2_refs)))
        l2_patents_df = self.expand_l2(pd.Series(list(l2_refs)))
        print('Shape of L2 expansion: {}'.format(l2_patents_df.shape))

        # Get all publication numbers from Seed, L1, and L2
        ## for use in getting anti-seed
        all_pub_nums = pd.Series(seed_df.publication_number) \
            .append(l1_patents_df.publication_number) \
            .append(l2_patents_df.publication_number)
        seed_and_expansion_pub_nums = set()
        for pub_num in all_pub_nums:
            seed_and_expansion_pub_nums.add(pub_num)
        print('Size of union of [Seed, L1, and L2]: {}'.format(len(seed_and_expansion_pub_nums)))

        # get the anti-seed set!
        anti_seed_df = self.anti_seed(pd.Series(list(seed_and_expansion_pub_nums)))
        
        # Get all publication numbers from Seed, Antiseed, L1, and L2
        full_pub_nums = pd.Series(seed_df.publication_number) \
            .append(l1_patents_df.publication_number) \
            .append(l2_patents_df.publication_number) \
            .append(anti_seed_df.publication_number)
        seed_antiseed_expansion_pub_nums = set()
        for pub_num in full_pub_nums:
            seed_antiseed_expansion_pub_nums.add(pub_num)
        print('Size of union of [Seed, Antiseed, L1, and L2]: {}'.format(len(seed_antiseed_expansion_pub_nums)))

        # get the remaining set!
        remain_patents_df = self.remain_patents(pd.Series(list(seed_antiseed_expansion_pub_nums)))

        return seed_df, l1_patents_df, l2_patents_df, anti_seed_df, remain_patents_df


    def derive_training_data_from_seeds(self):
        '''
        '''
        seed_df, l1_patents_df, l2_patents_df, anti_seed_df, remain_patents_df = \
            self.do_full_expansion()
        training_publications_df = \
            seed_df.append(anti_seed_df)[['publication_number', 'ExpansionLevel']]

        print('Loading training data text from {} publication numbers'.format(training_publications_df.shape))
        training_data_df = self.load_training_data_from_pubs(training_publications_df[['publication_number']])

        print('Merging labels into training data.')
        training_data_full_df = training_data_df.merge(training_publications_df, on=['publication_number'])

        return training_data_full_df, seed_df, l1_patents_df, l2_patents_df, anti_seed_df, remain_patents_df

    def load_from_disk_or_do_expansion(self):
        """Loads data for seed from disk, else derives/persists, then returns it.
        Checks for cached expansions for the given self.seed_name, and if a
        previous run is available it will load it from disk and return it;
        otherwise, it does L1 and L2 expansions, persists it in a cached
        'data/[self.seed_name]/' directory, and returns the data to the caller.
        """

        landscape_data_path = os.path.join(self.seed_data_path, 'landscape_data.pkl')

        if not os.path.exists(landscape_data_path):
            if not os.path.exists(self.seed_data_path):
                os.makedirs(self.seed_data_path)

            print('Loading landscape data from BigQuery.')
            training_data_full_df, seed_df, l1_patents_df, l2_patents_df, anti_seed_df, remain_patents_df = \
                self.derive_training_data_from_seeds()

            print('Saving landscape data to {}.'.format(landscape_data_path))
            with open(landscape_data_path, 'wb') as outfile:
                pickle.dump(
                    (training_data_full_df, seed_df, l1_patents_df, l2_patents_df, anti_seed_df, remain_patents_df),
                    outfile)
        else:
            print('Loading landscape data from filesystem at {}'.format(landscape_data_path))
            with open(landscape_data_path, 'rb') as infile:

                landscape_data_deserialized = pickle.load(infile)

                training_data_full_df, seed_df, l1_patents_df, l2_patents_df, anti_seed_df, remain_patents_df = \
                    landscape_data_deserialized

        self.training_data_full_df = training_data_full_df
        self.seed_df = seed_df
        self.l1_patents_df = l1_patents_df
        self.l2_patents_df = l2_patents_df
        self.anti_seed_df = anti_seed_df
        self.remain_patents_df = remain_patents_df

        return training_data_full_df, seed_df, l1_patents_df, l2_patents_df, anti_seed_df, remain_patents_df

    def load_all_utility_patents_for_inference(self):

        inference_data_query = '''
            SELECT DISTINCT *
            FROM
              `landscaping-318416.patents.utility_patents` AS b
            ;
        '''

        print('Loading patent texts from provided publication numbers.')
        #print('Training data query:\n{}'.format(training_data_query))
        inference_data_df = gbq.read_gbq(
            query=inference_data_query,
            project_id=self.bq_project,
            dialect='standard',
            configuration = {'query': {'useQueryCache': True, 'allowLargeResults': True}})

        return inference_data_df

    def load_inference_data(self, publications_df=None):
        tmp_table = 'patents._tmp_inference'
        self.load_df_to_bq_tmp(df=publications_df, tmp_table=tmp_table)

        inference_data_query = '''
            SELECT DISTINCT *
            FROM
              `landscaping-318416.patents.utility_patents` AS b
            INNER JOIN `{}` AS tmp ON b.publication_number = tmp.publication_number
            ;
        '''.format(tmp_table)

        print('Loading patent texts from provided publication numbers.')
        #print('Training data query:\n{}'.format(training_data_query))
        inference_data_df = gbq.read_gbq(
            query=inference_data_query,
            project_id=self.bq_project,
            dialect='standard',
            configuration = {'query': {'useQueryCache': True, 'allowLargeResults': True}})

        return inference_data_df
    
    def load_remain_patents(self, expansion_series):
        self.load_df_to_bq_tmp(pd.DataFrame(expansion_series, columns=['pub_num']), self.remain_tmp_table)

        remain_query = '''
            SELECT DISTINCT *
            FROM
              `landscaping-318416.patents.utility_patents` AS b
            LEFT OUTER JOIN `{}` AS tmp ON b.publication_number = tmp.pub_num
            WHERE
            tmp.pub_num IS NULL
            ;
        '''.format(self.remain_tmp_table)

        #print('Anti-seed query:\n{}'.format(anti_seed_query))
        remain_patents_df = gbq.read_gbq(
            query=remain_query,
            project_id=self.bq_project,
            dialect='standard',
            configuration = {'query': {'useQueryCache': True, 'allowLargeResults': True}})

        return remain_patents_df



    def data_for_inference(self, dataset, train_data_util):
        if self.dataset is None:
            raise ValueError('No patents loaded yet. Run expansion first (e.g., load_from_disc_or_do_expansion)')

        inference_data_path = os.path.join(self.seed_data_path, 'inference_data.pkl')

        if not os.path.exists(inference_data_path):
            print('Loading inference data from BigQuery.')
            pub_nums = self.dataset[['publication_number']].reset_index(drop=True)

            texts = self.load_training_data_from_pubs(pub_nums)

            inference_data = pub_nums.merge(texts, how='left', on=['publication_number'])
            
            inference_data['text'] = inference_data[['title_text','abstract_text']].agg('. '.join, axis=1)
            inference_data['refs'] = inference_data['refs'].fillna('')
            inference_data['cpcs'] = inference_data['cpcs'].fillna('')
            inference_data = inference_data[['publication_number', 'text', 'refs', 'cpcs']]

            # encode the data using the training data util
            padded_abstract_embeddings, refs_one_hot, cpc_one_hot = \
                train_data_util.prep_for_inference(inference_data.text, inference_data.refs, inference_data.cpcs)

            print('Saving inference data to {}.'.format(inference_data_path))
            with open(inference_data_path, 'wb') as outfile:
                pickle.dump(
                    (inference_data, padded_abstract_embeddings, refs_one_hot, cpc_one_hot),
                    outfile)
        else:
            print('Loading inference data from filesystem at {}'.format(inference_data_path))
            with open(inference_data_path, 'rb') as infile:
                inference_data_deserialized = pickle.load(infile)

                inference_data, padded_abstract_embeddings, refs_one_hot, cpc_one_hot = \
                    inference_data_deserialized

        return inference_data, padded_abstract_embeddings, refs_one_hot, cpc_one_hot


