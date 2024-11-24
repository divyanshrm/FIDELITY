#!/usr/bin/env python
# coding: utf-8
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import ast
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing.preprocessing import strip_tags
import sentence_transformers
import transformers
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import pickle
from keybert import KeyBERT
import umap
import hdbscan
import spacy
import nltk
from nltk.corpus import stopwords
from collections import defaultdict
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN,KMeans
from sklearn.cluster import KMeans

nlp = spacy.load("en_core_web_sm")

#output dir is the LLM model path
output_dir="blue/falcon"
model_config="blue/falcon/models--upstage--SOLAR-0-70b-16bit/snapshots/cfd79568e72039d4f857a1f04d16232ad5da51f1"
bert_model = sentence_transformers.SentenceTransformer('all-mpnet-base-v2')
kw_model = KeyBERT(model='all-mpnet-base-v2')

#only for dataframes having id list
def index_to_id(ids):
	"""Produces an index to id mapping which is used during the final output building phase
	Keyword arguments:
	:ids: A list of all ids of documents which is indexed similarly to the text"""
	d={}
	for x in range(len(ids)):
		d[x]=ids[x]
	return d

def preprocess_clean(document):
	"""Converts document/data into strings of tokens.
	Keyword arguments:
	:param document: List with all the input data.
	:return: A function call to a predefined gensim function that preprocesses data
	"""
	return simple_preprocess(strip_tags(document), deacc=True)

def find_embeddings_using_transformers(data,model):
	"""Creates embeddings for each token in the list of tokens parameter using sentence_transformers .
	Keyword arguments:
	:param data: List of tokens retrieved from input document after preprocessing
	:return: embeddings for each token
	"""
	embeddings = model.encode(data, show_progress_bar=False)
	return embeddings	 

def embed(x1,model):
	"""Converts document to an embedding
	Keyword arguments:
	:x1: A string contaning document text
	:model: A sentence-bert transformer"""
	reference_vector = find_embeddings_using_transformers(x1,model)

	return reference_vector

def get_keywords(text,model, ngram_range=1,top=5):
	'''Gets Keywords from an iterable containing text
	:Keyword arguments: 
	:text: iterable contaning text'''
	topic_entities = model.extract_keywords(text,keyphrase_ngram_range=(1,ngram_range), stop_words='english',top_n=top,use_mmr=True, diversity=0.4)  

	return topic_entities

# * Scorers/Metrics
# from rouge_score import rouge_scorer
def get_vector_similarity(x1, x2):
	'''Computes cosine similarity between two vectors
	Keyword arguments: 
	:x1: A vector
	:x2: A vector'''
	similarity = torch.nn.functional.cosine_similarity(x1=x1, x2=x2, dim=-1)
	return similarity

def get_thresholded_keywords(text,model,threshold=0.0):
	'''Gets Keywords from a text iterable using Keybert, with a specific threshold on the cosine similarity of keywords with the document:
	 Keyword arguments: 
	:text: iterable contaning text
	:model: A Keybert model'''
	all_keywords = set([])
	keywords_per_doc=[]

	#for sample in tqdm(text):
	for sample in text:
		topics = get_keywords(sample, model,ngram_range=3,top=10)
		temp_topics=[]
		scores=[0]
		for x in topics:
			scores.append(x[1])
		threshold=max(scores)/3
		for x in topics:
			if x[1]>threshold:
				temp_topics.append(x[0])
		[all_keywords.add(x) for x in temp_topics]
		keywords_per_doc.append([x for x in temp_topics])

	return list(all_keywords),keywords_per_doc

def embed_keywords(all_keywords,model):
	'''Embeds all the keywords using Sentence bert: 
	Keyword Arguments:
	:all_keywords: iterable contaning extracted keywords
	:model: A Sentence-Bert model'''
	keyword_embedding_dict = {}

	all_keywords = list(all_keywords) # ORDER IS IMPORTANT to get embedding 
	keyword_embeddings=find_embeddings_using_transformers(all_keywords,model)
	for x in range(len(keyword_embeddings)):
		keyword_embedding_dict[all_keywords[x]]=keyword_embeddings[x]
	word_values=list(keyword_embedding_dict.values())
	return keyword_embedding_dict,word_values

def dimension_reduce(word_values,method="UMAP"):
	'''
	Reduces the Dimensions of Embeddings using UMAP/TSNE
	Keyword Arguments:
	:word_values:An iterable containing embeddings
	:method:UMAP or TSNE 'default UMAP'
	'''
	if method=="TSNE":
		out = TSNE(n_components=3, learning_rate='auto',init='random', perplexity=3).fit_transform(np.array(word_values))
	elif method=="UMAP":
		u_map = umap.UMAP(n_neighbors=25,n_components=5,metric='cosine',min_dist=0.0)
		out=u_map.fit_transform(np.array(word_values))
	return out,u_map

def clustering(embeddings,method='HDBSCAN'):
	'''
	Clustering Algorithm
	Keyword Arguments:
	:embeddings:An iterable containing embeddings
	:method:Clustering Algorithm to use, Default HDBSCAN, Recommended
	'''
	if method=='DBSCAN':
		clusterer=DBSCAN(eps=0.2, min_samples=15,metric='cosine',min_cluster_size=5, min_dist=0.0)
	elif method=='HDBSCAN':
		clusterer = hdbscan.HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
	elif method=='KMeans':
		clusterer=KMeans(n_clusters=10)
	cluster_labels = clusterer.fit_predict(embeddings)
	return clusterer,cluster_labels

#the directory to save the models

def initialize_llm(output_dir,model_config):
	config = AutoConfig.from_pretrained(model_config, trust_remote_code=True,cache_dir=output_dir)
	tokenizer = AutoTokenizer.from_pretrained(model_config,config = AutoConfig.from_pretrained(model_config, trust_remote_code=True,cache_dir=output_dir)
,cache_dir=output_dir)


	pipeline = transformers.pipeline(
		"text-generation",
		model=model_config,
		tokenizer=tokenizer,
		torch_dtype=torch.bfloat16,
		trust_remote_code=True,
	device_map="auto",
		config=config,model_kwargs={'cache_dir':output_dir},
	)
	return pipeline,tokenizer

def get_response(text,pipeline,tokenizer):
	'''
	Gets Response from LLAMA-2
	Keyword Arguments:
	:pipeline:Created pipeline object
	:tokenizer:LLAMA-2 Tokenizer
	'''
	
	input_text="### User:\n"+text+"\n\n### Assistant:\n"

	sequences = pipeline(
	input_text,
	max_length=5000,
	do_sample=True,
	return_full_text=False,
	top_k=10,
	num_return_sequences=1,
	eos_token_id=tokenizer.eos_token_id,
	pad_token_id=tokenizer.eos_token_id
)
		  
	return sequences[0]['generated_text']

#Build function to retrieve generalized topic responses

def get_semantic_label(keystring,pipeline,tokenizer):
	'''
	Preprocesses the input and generates the prompt for LLAMA-2
	Keyword Arguments:
	:keystring:Iterable containing keywords clustered together
	:pipeline:Created pipeline object
	:tokenizer:Falcon40 Tokenizer
	'''
	prompt = f"Combine the following keywords into a general topic : {keystring}. Only reply with a brief single phrase which is the topic and nothing else. Your reply should not be more than 6 words."
	response = get_response(prompt,pipeline,tokenizer)
	filter = ''.join([chr(i) for i in range(1, 32)])
	filtered=response.translate(str.maketrans('', '', filter))

	return filtered

def check(s):
	'''
	checks if the output fits the desired format.
	keyword arguments:
	:s:A string output
	'''
	if s.isspace() or len(s)==0:
		return True
	if 'topic' in s:
		return True
	if 'Topic' in s:
		return True
	if len(s.split(' '))>9:
		return True
	if len(s)<=3 or (len(s.split(' '))<2):
		return True
	return False

def clean(string):
	"""
	Removes double quotes, forward slashes, and full stops (", / , .") from a given string.
	keyword arguments:
	:string: A string to preprocess
	"""
	temp=string.replace('"', '')
	temp2=temp.replace("/",'')
	temp3=temp2.replace(".",'')
	return temp3.strip()
def truncate_to_900_tokens(text):
    """
    Truncates the input text to a maximum of 900 tokens.
    
    Args:
    - text (str): The input string to be truncated.
    
    Returns:
    - str: The truncated string.
    """
    # Split the text into tokens (words)
    tokens = text.split()
    
    # Limit to 900 tokens
    truncated_tokens = tokens[:900]
    
    # Reconstruct the truncated string
    truncated_text = ' '.join(truncated_tokens)
    
    return truncated_text
def get_topics_from_keywords(cluster_labels,pipeline,tokenizer,all_keywords):
	"""
	Creates and dispatches prompts to falcon 40b from clustered keywords
	keyword arguments:
	:cluster labels: Cluster labels from clusterer
	:pipeline:Created pipeline object
	:tokenizer:Falcon40 Tokenizer
	:all keywords: Iterable containing all keywords extracted from documents
	"""
	label_to_semantic_topic = {}
	keyword_to_semantic_topic={}

	#for label in tqdm(np.unique(cluster_labels)):
	for label in np.unique(cluster_labels):
		class_indeces = np.where(cluster_labels == label)
		topic_keywords = np.asarray(list(all_keywords))[class_indeces]
		if label != -1:
			keystring = truncate_to_900_tokens(",".join(topic_keywords[:50]))
			response = get_semantic_label(keystring,pipeline,tokenizer)
			c=0
			while check(response) and c<=3 :
				response=get_semantic_label(keystring,pipeline,tokenizer)
				c+=1
			cleaned=clean(response)
			for key in topic_keywords:
				keyword_to_semantic_topic[key]=cleaned		 
			label_to_semantic_topic[label] = cleaned

	label_to_semantic_topic[-1]="Outlier"

	return label_to_semantic_topic,keyword_to_semantic_topic

def get_topics_per_doc(keywords_per_doc,keyword_to_semantic_topic,model,threshold,doc_embeddings):
	"""
	Generates topics for each document after applying a similarity threshold between the identified topic and the document
	keyword arguments:
	:keywords_per_doc: Dictionary Mapping from Keywords to Document
	:keyword_to_semantic_topic:Dictionary mapping from keywrod to extracted topic
	:model:Sentence Bert Model
	:threshold: Threshold value for similarity. Between 0 and 1.
	:doc_embeddings:Iterable containing document embeddings for all documents
	"""
	all_topics=[]
	i=0
	#for x in tqdm(keywords_per_doc):
	for x in keywords_per_doc:
		topics=[]
		for y in x:
			if y in keyword_to_semantic_topic:
				topic=keyword_to_semantic_topic[y]
				if topic not in topics:
					score=get_vector_similarity(torch.tensor(embed(topic,model)),torch.tensor(doc_embeddings[i]))
					if score>threshold:
						topics.append(topic)
		i+=1
		all_topics.append(topics)
	return all_topics

def get_ids(df,index_to_id_mapping):
	'''
	This func returns a dictionary which contains topics to Docs mapping
	keyword arguments:
	:df: Generated Dataframe containing topic-doc mapping'''
	topic_doc=defaultdict(list)
	for x in range(len(df.text)):
		for y in df.Topics[x]:
			topic_doc[y].append(index_to_id_mapping[x])

	return topic_doc

def get_topic_keyword_mapping(all_keywords,keyword_to_semantic_topic):
	'''
	This func returns a dictionary which contains topics to Keyword mapping
	keyword arguments:
	:df: Generated Dataframe containing topic-doc mapping'''
	topic_key=defaultdict(list)
	for x in all_keywords:
		if x in keyword_to_semantic_topic and x not in topic_key[keyword_to_semantic_topic[x]]:
			topic_key[keyword_to_semantic_topic[x]].append(x)

	return topic_key

def get_len(x):
	'''
	This func returns a dictionary which contains topics to Docs mapping
	keyword arguments:
	:df: Generated Dataframe containing topic-doc mapping'''
	l=[]
	for y in x:
		l.append(len(y))

	return l

def get_keywords_from_topic_list(x,topic_key):
	l=[]
	for y in x:
		l.append(topic_key[y])

	return l

def get_output_df(all_topics,keywords_per_doc,text,index_to_id_mapping,all_keywords,keyword_to_semantic_topic):
	new_df=pd.DataFrame()
	new_df['Topics']=all_topics
	new_df['Keywords']=keywords_per_doc
	new_df['text']=text
	topic_key=get_topic_keyword_mapping(all_keywords,keyword_to_semantic_topic)
	topic_doc=get_ids(new_df,index_to_id_mapping)
	output_df=pd.DataFrame()
	output_df['Topics']=[x for x in topic_doc.keys()]
	output_df['Documents']=[x for x in topic_doc.values()]
	output_df['Number of Documents']=get_len(output_df['Documents'])
	output_df['Topic_Cluster']=get_keywords_from_topic_list(output_df['Topics'],topic_key)
	sorted_df=output_df.sort_values('Number of Documents',ascending=False).reset_index().drop('index',axis=1)[['Documents','Topics','Topic_Cluster']]

	return sorted_df,topic_key

def collapse(df,threshold):
	if threshold==1:
		mapping = {}
		for x in df.Topics:
			mapping[x]=x
		return df, mapping
	df=df.copy()

	def group_similar_sentences(sentences,threshold=0.85):
		#bert_model = sentence_transformers.SentenceTransformer('all-mpnet-base-v2')
		def find_cos(bert_model,x1,x2):
			a=find_embeddings_using_transformers(x1,bert_model)
			b=find_embeddings_using_transformers(x2,bert_model)
			return get_vector_similarity(torch.tensor(a),torch.tensor(b))

		groups = []
		grouped_indices = set()
		emb=torch.tensor(find_embeddings_using_transformers(sentences,bert_model))
		#for i in tqdm(range(len(sentences))):
		for i in range(len(sentences)):
			if i not in grouped_indices:
				similar_group = [sentences[i]]
				for j in range(i + 1, len(sentences)):
					if j not in grouped_indices:
						similarity_score = get_vector_similarity(emb[i],emb[j])
						if similarity_score >= threshold:
							similar_group.append(sentences[j])
							grouped_indices.add(j)
				groups.append(similar_group)

		return groups

	sentences = df.Topics

	similar_groups = group_similar_sentences(sentences, threshold)
	mapping={}
	for x in similar_groups:
		for y in x:
			mapping[y]=x[0]
	for x in range(len(df.Topics)):
		df.Topics[x]=mapping[df.Topics[x]]

	def merge_duplicate_topics(df):
		# Define custom aggregation function to concatenate lists
		def concat_lists(x):
			concatenated_list = []
			for sublist in x:
				concatenated_list.extend(sublist)
			return concatenated_list

		# Grouping by the specified column and aggregating other columns
		grouped_df = df.groupby('Topics').agg({col: concat_lists for col in df.columns if col != 'Topics'}).reset_index()
		for x in range(len(grouped_df)):

			grouped_df['Documents'][x]=list(np.unique(np.asarray(grouped_df['Documents'][x])))

		def sort_df_by_list_length(df, column_name):
			# Sort DataFrame based on the length of lists in the specified column
			sorted_df = df.iloc[df[column_name].apply(len).argsort()[::-1]].reset_index(drop=True)
			return sorted_df
		
		return sort_df_by_list_length(grouped_df, 'Documents')[['Documents','Topics','Topic_Cluster']]

	out=merge_duplicate_topics(df)

	return out,mapping

def transform_df(df,mapping):
	def merge_duplicate_topics(df):
		# Define custom aggregation function to concatenate lists
		def concat_lists(x):
			concatenated_list = []
			for sublist in x:
				concatenated_list.extend(sublist)
			return concatenated_list

		# Grouping by the specified column and aggregating other columns
		grouped_df = df.groupby('Topics').agg({col: concat_lists for col in df.columns if col != 'Topics'}).reset_index()

		for x in range(len(grouped_df)):

			grouped_df['Documents'][x]=list(np.unique(np.asarray(grouped_df['Documents'][x])))

		def sort_df_by_list_length(df, column_name):
			# Sort DataFrame based on the length of lists in the specified column
			sorted_df = df.iloc[df[column_name].apply(len).argsort()[::-1]].reset_index(drop=True)
			return sorted_df
		
		return sort_df_by_list_length(grouped_df, 'Documents')[['Documents','Topics','Topic_Cluster']]
	
	for x in range(len(df.Topics)):
		df['Topics'][x]=mapping[df['Topics'][x]]
	
	out=merge_duplicate_topics(df)

	return out
	
def topic_modelling(docs,ids,pipeline,tokenizer,threshold):
	text = [' '.join(preprocess_clean(str(doc))) for doc in docs]
	index_to_id_mapping=index_to_id(ids)
	#bert_model = sentence_transformers.SentenceTransformer('all-mpnet-base-v2')
	doc_embeddings=find_embeddings_using_transformers(text,bert_model)
	#kw_model = KeyBERT(model='all-mpnet-base-v2')
	all_keywords,keywords_per_doc=get_thresholded_keywords(text,kw_model)
	keyword_embedding_dict,word_values=embed_keywords(list(all_keywords),bert_model)
	out,reduce=dimension_reduce(np.asarray(word_values),"UMAP")
	clusterer,cluster_labels=clustering(np.array(out),'HDBSCAN')
	   
	label_to_semantic_topic,keyword_to_semantic_topic=get_topics_from_keywords(cluster_labels,pipeline,tokenizer,all_keywords)
	all_topics=get_topics_per_doc(keywords_per_doc,keyword_to_semantic_topic,bert_model,0.4,doc_embeddings)
	odf,topic_key=get_output_df(all_topics,keywords_per_doc,text,index_to_id_mapping,all_keywords,keyword_to_semantic_topic)

	return (odf,topic_key),label_to_semantic_topic,clusterer,reduce

class Topic_Modelling():
    def __init__(self, pipeline, tokenizer, resource_path="resources", scenario="lith"):
        if not os.path.isdir(resource_path):
            os.mkdir(resource_path)
        self.resource_path = os.path.join(resource_path, scenario)
        self.scenario = scenario
        if not os.path.isdir(self.resource_path):
            os.mkdir(self.resource_path)
        self.resource_built = self.check_resources()
        self.pipeline, self.tokenizer = pipeline, tokenizer

    def resource_building(self, docs_ids, redo=False, threshold=0.85):
        if self.resource_built and (not redo) and self.check_resources():
            self.topic_to_keywords, self.clusterer, self.umap_module, self.label_to_semantic_topic, self.output_df, self.mapping = self.load_resources()
            odf= transform_df(self.output_df, self.mapping)
            odf['Scenario'] = self.scenario
            return odf
        docs = []
        ids = []
        for x in docs_ids:
            ids.append(x['doc_id'])
            docs.append(x['text_english'])
        (self.output_df, self.topic_to_keywords), self.label_to_semantic_topic, self.clusterer, self.umap_module = topic_modelling(docs, ids, self.pipeline, self.tokenizer, threshold)
        self.resource_built = True
        odf, self.mapping = collapse(self.output_df, threshold)
        self.save_resources()
        odf['Scenario'] = self.scenario
        return odf

    def predict(self, docs_ids, threshold=0.85):
        doc_id = docs_ids['doc_id']
        doc = docs_ids['text_english']
        if self.resource_built == False:
            raise Exception("Resources have not been built")
        text = [' '.join(preprocess_clean(str(doc)))]
        doc_embeddings = find_embeddings_using_transformers(text, bert_model)
        all_keywords, _ = get_thresholded_keywords(text, kw_model)
        if len(all_keywords)==0:
            return None
        keyword_embedding_dict, word_values = embed_keywords(list(all_keywords), bert_model)
        out = self.umap_module.transform(word_values)
        detected_topics = set()
        for x in hdbscan.approximate_predict(self.clusterer, out)[0]:
            if x != -1:
                temp_topic = self.label_to_semantic_topic[x]
                
                score = get_vector_similarity(torch.tensor(embed(temp_topic, bert_model)), torch.tensor(embed(text, bert_model)))
                if score >= 0.4 and temp_topic in self.mapping:
                    detected_topics.add(temp_topic)
        all_topics = list(detected_topics)
        keywords_per_topic = []
        for topic in all_topics:
            keywords_per_topic.append(self.topic_to_keywords[topic])
        output_df = pd.DataFrame()
        dataframe_ids = []
        for x in range(len(all_topics)):
            dataframe_ids.append([doc_id])
        output_df['Documents'] = dataframe_ids
        output_df['Topics'] = all_topics
        output_df['Topic_Cluster'] = keywords_per_topic
        
        return transform_df(output_df, self.mapping)

    def save_resources(self):
        if not self.resource_built:
            raise Exception("Resources have not been built")
        topic_to_keywords_path = os.path.join(self.resource_path, 'topic_to_keywords.pkl')
        clusterer_path = os.path.join(self.resource_path, 'clusterer.pkl')
        umap_module_path = os.path.join(self.resource_path, 'umap_module.pkl')
        label_to_semantic_topic_path = os.path.join(self.resource_path, 'label_to_semantic_topic.pkl')
        output_df_path = os.path.join(self.resource_path, 'output_df.pkl')
        mapping_path = os.path.join(self.resource_path, 'mapping.pkl')
        with open(clusterer_path, 'wb') as file:
            pickle.dump(self.clusterer, file)
        with open(topic_to_keywords_path, 'wb') as file:
            pickle.dump(self.topic_to_keywords, file)
        with open(umap_module_path, 'wb') as file:
            pickle.dump(self.umap_module, file)
        with open(label_to_semantic_topic_path, 'wb') as file:
            pickle.dump(self.label_to_semantic_topic, file)
        with open(output_df_path, 'wb') as file:
            pickle.dump(self.output_df, file)
        with open(mapping_path, 'wb') as file:
            pickle.dump(self.mapping, file)

    def check_resources(self):
        topic_to_keywords_path = os.path.join(self.resource_path, 'topic_to_keywords.pkl')
        clusterer_path = os.path.join(self.resource_path, 'clusterer.pkl')
        umap_module_path = os.path.join(self.resource_path, 'umap_module.pkl')
        label_to_semantic_topic_path = os.path.join(self.resource_path, 'label_to_semantic_topic.pkl')
        output_df_path = os.path.join(self.resource_path, 'output_df.pkl')
        mapping_path = os.path.join(self.resource_path, 'mapping.pkl')
        flag1 = os.path.isdir(self.resource_path)
        flag2 = os.path.isfile(topic_to_keywords_path) and os.path.isfile(clusterer_path) and os.path.isfile(umap_module_path) and os.path.isfile(label_to_semantic_topic_path) and os.path.isfile(output_df_path)
        flag3=os.path.isfile(mapping_path)
        if flag2 and not flag3:
            with open(output_df_path, 'rb') as file:
                output_df = pickle.load(file)
            mapping={}
            for x in output_df.Topics:
                mapping[x]=x
            
            with open(mapping_path, 'wb') as file:
                pickle.dump(mapping, file)
        is_built = flag1 and flag2
        self.resource_built = is_built
        print("is_built flag value, checking if the resources for the given scenario exist: ", is_built)
        return is_built

    def load_resources(self):
        if (not self.check_resources()) or (not self.resource_built):
            raise Exception("Resources have not been built")
        topic_to_keywords_path = os.path.join(self.resource_path, 'topic_to_keywords.pkl')
        clusterer_path = os.path.join(self.resource_path, 'clusterer.pkl')
        umap_module_path = os.path.join(self.resource_path, 'umap_module.pkl')
        label_to_semantic_topic_path = os.path.join(self.resource_path, 'label_to_semantic_topic.pkl')
        output_df_path = os.path.join(self.resource_path, 'output_df.pkl')
        mapping_path = os.path.join(self.resource_path, 'mapping.pkl')
        with open(topic_to_keywords_path, 'rb') as file:
            topic_to_keywords = pickle.load(file)
        with open(clusterer_path, 'rb') as file:
            clusterer = pickle.load(file)
        with open(umap_module_path, 'rb') as file:
            umap_module = pickle.load(file)
        with open(label_to_semantic_topic_path, 'rb') as file:
            label_to_semantic_topic = pickle.load(file)
        with open(output_df_path, 'rb') as file:
            output_df = pickle.load(file)
        with open(mapping_path, 'rb') as file:
            mapping = pickle.load(file)
        return topic_to_keywords, clusterer, umap_module, label_to_semantic_topic, output_df, mapping
    def collapse_topics(self,threshold):
        df,self.mapping=collapse(self.output_df,threshold)
        self.save_resources()
        return df
def get_sentiment_sentences_online_processing(texts, indexes, module, threshold=0.85):
    """gives a json with sentences containing the keywords from the topic cluster
    params:
    :texts: a python list containing the text
    :indexes: a python list containing the identifier (ID) to the documents as in the output dataframe of the topic modelling module
    :df: output df from topic modelling
    """
    #from transformers import pipeline

    def get_matched(texts):
        # Preprocess the texts
        texts = [' '.join(preprocess_clean(str(text))) for text in texts]

        # Get all keywords and keywords per document
        all_keywords, keywords_per_doc = get_thresholded_keywords(texts, kw_model)

        # Flatten the keywords_per_doc to get a list of all keywords
        all_doc_keywords = [keyword for keywords in keywords_per_doc for keyword in keywords]

        # Embed all unique keywords
        unique_keywords = list(set(all_doc_keywords))
        keyword_embedding_dict, word_values = embed_keywords(unique_keywords, bert_model)

        # Transform all keyword embeddings using UMAP
        transformed_embeddings = module.umap_module.transform(word_values)
        transformed_dict = {kw: transformed_embeddings[i] for i, kw in enumerate(unique_keywords)}

        # Predict clusters for all transformed embeddings in batch
        cluster_output = hdbscan.approximate_predict(module.clusterer, transformed_embeddings)[0]

        # Create a dictionary to map each keyword to its cluster
        keyword_to_cluster = {unique_keywords[i]: cluster_output[i] for i in range(len(unique_keywords))}

        # Gather all unique topic labels
        unique_topic_labels = list(set(
            module.label_to_semantic_topic[cluster_label] 
            for cluster_label in set(keyword_to_cluster.values()) if cluster_label != -1
        ))

        # Pre-embed all topics in a single batch
        topic_embeddings = embed(unique_topic_labels, bert_model)
        topic_embedding_dict = {
            unique_topic_labels[i]: torch.tensor(topic_embeddings[i]) 
            for i in range(len(unique_topic_labels))
        }

        # Pre-embed all texts in a single batch and convert to tensors
        text_embeddings = embed(texts, bert_model)
        text_embedding_tensors = torch.tensor(text_embeddings)

        matched_topics = []
        for doc_idx, doc_keywords in enumerate(keywords_per_doc):
            matched_topic = defaultdict(list)

            for keyword in doc_keywords:
                if keyword in keyword_to_cluster and keyword_to_cluster[keyword] != -1:
                    temp_topic = module.label_to_semantic_topic[keyword_to_cluster[keyword]]
                    temp_topic_embedding = topic_embedding_dict[temp_topic]
                    text_embedding = text_embedding_tensors[doc_idx]

                    score = get_vector_similarity(temp_topic_embedding, text_embedding)

                    if score >= 0.4:
                        matched_topic[temp_topic].append(keyword)

            matched_topics.append(matched_topic)

        return matched_topics




    def batch_analyze_sentiment(texts):
        # Load the sentiment analysis model
        sentiment_model = transformers.pipeline("text-classification", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

        # Perform sentiment analysis on the batch of texts
        results = sentiment_model(texts)

        # Extract positive and negative sentiments with their scores

        return results

    #def preprocess_clean(document):
    #    """Converts document/data into strings of tokens.
    #    Keyword arguments:
    #    :param document: List with all the input data.
    #    :return: A function call to a predefined gensim function that preprocesses data
    #    """
    #    return simple_preprocess(strip_tags(document), deacc=True)

    t_c = defaultdict(list)
    d_t = defaultdict(list)
    #bert_model = sentence_transformers.SentenceTransformer('all-mpnet-base-v2')
    keywords_per_doc = {}
    #kw_model = KeyBERT(model='all-MiniLM-L6-v2')
    mapping = module.mapping
    #print("### Matching documents to topics ###")
    temps = get_matched(texts)         
    for x in range(len(texts)):
        #temp=get_matched(texts[x])         
        temp = temps[x]
        d_t[indexes[x]] = list(temp.keys())
        topics = defaultdict(list)
        for x in temp:
            if x in mapping.keys():
                topics[mapping[x]] += temp[x]
        for y in topics.keys():
            for z in topics[y]:
                t_c[y].append(z)

    stop_words = set(stopwords.words("english"))
    def remove_stopwords(text, stop_words):
        word_tokens = nltk.word_tokenize(text)
        filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
        return ' '.join(filtered_text)

    d_s = defaultdict(list)

    #nlp = spacy.load("en_core_web_sm")
    stopwords_to_plain = {}
    #for x in tqdm(range(len(indexes))):
    for x in range(len(indexes)):
        doc = nlp(texts[x])
        temp = [sent.text for sent in doc.sents]

        for y in temp:
            preprocessed = remove_stopwords(' '.join(preprocess_clean(str(y))[:100]), stop_words)
            stopwords_to_plain[preprocessed] = y
            d_s[indexes[x]].append(preprocessed)
            

    final = defaultdict(list)
    required = []
    t = 0
    #for x in tqdm(indexes):
    #print("### Getting sentences for topics ###")
    for x in indexes:
        if x in d_t:
            final["Documents"].append({})
            final["Documents"][-1]["ID"] = str(x)
            final["Documents"][-1]["Topics"] = []               
            for y in d_t[x]:
                final["Documents"][-1]["Topics"].append({"Topic": y, "Sentences": []})

                hashmap = {}
                for a in d_s[x]:
                    if a not in hashmap:
                        hashmap[a] = 1
                        keywords_matched = []

                        for z in t_c[y]:
                            if z in a:
                                keywords_matched.append(z)

                        if keywords_matched:
                            required.append(a)
                            final["Documents"][-1]["Topics"][-1]["Sentences"].append({"text": a, "Sentiment": "", "Score": -1, "Keywords": keywords_matched})    
                if not final["Documents"][-1]["Topics"][-1]["Sentences"]:
                    final["Documents"][-1]["Topics"].pop()

    s_sentiment = {}
    results=batch_analyze_sentiment(required)
    #for x in tqdm(range(len(required))):
    for x in range(len(required)):
        s_sentiment[required[x]]=(results[x]['label'],results[x]['score'])
        #s_sentiment[required[x]] = ("N/A", 0.999)

    c = 0
    for x in final["Documents"]:
        for y in x["Topics"]:
            for z in y["Sentences"]:
                c += 1
                z["Sentiment"] = s_sentiment[z['text']][0]
                z["Score"] = s_sentiment[z['text']][1]
                z["text"] = stopwords_to_plain[z['text']]


    return dict(final)
def get_info_packets(text,indexes,module,info_source,threshold=0.7):
    output_json=k
    id_info={}
    
    for x in range(len(indexes)):
        id_info[str(indexes[x])]=str(info_source[x])
    
    topic_id={}
    for i in range(len(module.output_df)):
        #topic_id[topic_id_df['name'][i].strip()]=topic_id_df['topic_id'][i]
        topic_id[module.output_df.Topics[i].strip()]=i+1000
    d=[]
    t=[]
    
    scores=[]
    div=[]
    info_source=[]
    tid=[]
    iid=[]
    checking=[]
    c=0
    def get_score(score):
        if score>0:
            flag=1
        elif score<0:
            flag=-1
        s=abs(score)
        if s<0.7:
            return 0.0
        if s>=0.7 and s<0.85:
            return flag*1.0
        if s>=0.85 and s<0.95:
            return flag*2.0
        elif s>=0.95:
            return flag*3.0


    for x in output_json['Documents']:
        for y in x['Topics']:
            for z in y['Sentences']:
                d.append(x['ID'])
                t.append(y['Topic'])
                if z['Sentiment']=='POSITIVE':
                    scores.append(z['Score'])
                elif z['Sentiment']=='NEGATIVE':
                    scores.append(-z['Score'])
                div.append(get_score(scores[-1]))
                info_source.append(str(id_info[str(x['ID'])]))
                tid.append(str(topic_id[y['Topic'].strip()]))
                iid.append(c)
                checking.append(z['Sentiment'])
                c+=1
    import pandas as pd

    def drop_rows_with_zero(df, column_name):
        """
        Drops rows in a DataFrame where a specified column has a value of 0.

        Args:
        - df (pandas.DataFrame): The DataFrame from which rows are to be dropped.
        - column_name (str): The name of the column to check for zero values.

        Returns:
        - pandas.DataFrame: The DataFrame with rows dropped where the specified column has a value of 0.
        """
        return df[df[column_name] != 0.0]
    output=pd.DataFrame({'Info Packet ID':iid,'Information Source':info_source,'Document ID':d,'Topic ID':tid,'Topic String':t,'Stance':div})
    final=drop_rows_with_zero(output,'Stance')
    return final
