import os
from argparse import ArgumentParser

# Third parties
import numpy as np

# Import from other modules
from tagger.data.tools import load_data, to_ML, select_events
from tagger.model.common import fromFolder, fromYaml
from tagger.plot.basic import basic, plot_event_ROC, plot_PCA, plot_latent,plot_nontrained_event_ROC,plot_latent_vs_variable,plot_output_scores
import tagger.plot.style as style 

from sklearn.model_selection import train_test_split

from argparse import ArgumentParser

from sklearn.decomposition import PCA
import itertools

import matplotlib.pyplot as plt
import mplhep as hep


parser = ArgumentParser()
    # Training argument
parser.add_argument(
        '-o', '--output', default='output/ds', help='Output model directory path, also save evaluation plots'
)

parser.add_argument(
        '-i', '--input', default='event_training_data', help='input training data'
    )

args = parser.parse_args()

training_data = args.input

event_labels = {'minbias' :0,'TT_PU200' : 1,'HH_4b' : 2}
event_label_style = {'minbias':'Minbias', 'TT_PU200': '$t\\overline{t}$', 'HH_4b':'HH->bbbb'}

jet_class_labels_vector = []
jet_class_predictions_vector = []
jet_pt_predictions_vector = []
jet_embeddings_vector = []
candidate_features_vector = []
jet_features_vector = []
event_label_vector = []
bonus_event_info_vector = []

for event_label in event_labels.keys():

    jet_class_labels = np.load(training_data+'/'+event_label+'/jet_class_labels.npy')
    jet_class_predictions = np.load(training_data+'/'+event_label+'/jet_class_predictions.npy')
    jet_pt_predictions = np.load(training_data+'/'+event_label+'/jet_pt_predictions.npy')
    jet_embeddings = np.load(training_data+'/'+event_label+'/jet_embeddings.npy')
    candidate_features = np.load(training_data+'/'+event_label+'/candidate_features.npy')
    jet_features = np.load(training_data+'/'+event_label+'/jet_features.npy')
    event_info = np.load(training_data+'/'+event_label+'/bonus_event_info.npy')
    event_label = np.load(training_data+'/'+event_label+'/event_label.npy')
    
    jet_class_labels_vector.append(jet_class_labels)
    jet_class_predictions_vector.append(jet_class_predictions)
    jet_pt_predictions_vector.append(jet_pt_predictions)
    jet_embeddings_vector.append(jet_embeddings)
    candidate_features_vector.append(candidate_features)
    jet_features_vector.append(jet_features)
    event_label_vector.append(event_label)
    bonus_event_info_vector.append(event_info)
    
jet_class_labels_vector = np.concatenate(jet_class_labels_vector, axis=0)
jet_class_predictions_vector = np.concatenate(jet_class_predictions_vector, axis=0)
jet_pt_predictions_vector = np.concatenate(jet_pt_predictions_vector, axis=0)
jet_embeddings_vector = np.concatenate(jet_embeddings_vector, axis=0)
candidate_features_vector = np.concatenate(candidate_features_vector, axis=0)
jet_features_vector = np.concatenate(jet_features_vector, axis=0)
bonus_event_info_vector = np.concatenate(bonus_event_info_vector, axis=0)
event_label_vector = np.concatenate(event_label_vector, axis=0)

event_ids = np.arange(0,len(event_label_vector))

train_indices, test_indices = train_test_split(event_ids,test_size=0.2,random_state=42)

jet_class_labels_vector = jet_class_labels_vector[test_indices]
jet_class_predictions_vector =jet_class_predictions_vector[test_indices]
jet_pt_predictions_vector = jet_pt_predictions_vector[test_indices]
jet_embeddings_vector = jet_embeddings_vector[test_indices]
candidate_features_vector = candidate_features_vector[test_indices]
jet_features_vector = jet_features_vector[test_indices]
event_label_vector = event_label_vector[test_indices]
bonus_event_info_vector = bonus_event_info_vector[test_indices]

y =  np.eye(len(event_labels.keys()))[np.asarray(event_label_vector, dtype=int)]

selected_events = select_events(jet_features_vector)

model = fromFolder(args.output)

#X = np.concatenate([jet_features_vector,jet_class_predictions_vector,jet_pt_predictions_vector[..., np.newaxis]],axis=2)
X = candidate_features_vector
print('jet_features: ',jet_features_vector)
print('jet_class: ',jet_class_predictions_vector)
print('jet_pt: ',jet_pt_predictions_vector)
print('jet_features_shape: ',jet_features_vector.shape)
print('jet_class_shape: ',jet_class_predictions_vector.shape)
print('jet_pt_shape: ',jet_pt_predictions_vector.shape)
print('X: ',X)
print('X_shape: ',X.shape)

batch_size = X.shape[0]
X_flattened = X.reshape(X.shape[0]*X.shape[1],X.shape[2],X.shape[3])

y_jet_pred_flattened = model.predict(X_flattened)[0]
flattened_jet_embeddings = model.embedding_predict(X_flattened)
print('y_pred: ',y_jet_pred_flattened)
print('y_pred_shape: ',y_jet_pred_flattened.shape)

plot_dir = args.output + '/plots/testing_samples'
os.makedirs(plot_dir, exist_ok=True)

y_jet_pred = y_jet_pred_flattened.reshape(batch_size,12,y_jet_pred_flattened.shape[-1])
jet_embeddings = flattened_jet_embeddings.reshape(batch_size,12,flattened_jet_embeddings.shape[-1])

flattened_jet_classes = jet_class_labels_vector.reshape(-1, jet_class_labels_vector.shape[-1])

summed_classes = np.sum(flattened_jet_classes,axis=1)
non_zero_jets = np.argwhere(summed_classes != 0 )

non_zero_flattened_jet_embeddings = np.squeeze(flattened_jet_embeddings[non_zero_jets])

pca = PCA(n_components=2)
jet_principle_components = pca.fit_transform(non_zero_flattened_jet_embeddings)

jet_xmax, jet_xmin = np.percentile(jet_principle_components[:,0],99.9), np.percentile(jet_principle_components[:,0],0.01)
jet_ymax, jet_ymin = np.percentile(jet_principle_components[:,1],99.9), np.percentile(jet_principle_components[:,1],0.01)

tiled_event_class_labels = np.repeat(y, 12, axis=0) 
non_zero_tiled_event_classes = np.squeeze(tiled_event_class_labels[non_zero_jets])
non_zero_flattened_jet_classes = np.squeeze(flattened_jet_classes[non_zero_jets])

print("===== plot jet embeddings ====")
print(" plot jet pca per jet class" )
plot_PCA(jet_principle_components, np.argmax(non_zero_flattened_jet_classes,axis=1),((jet_xmin, jet_xmax),(jet_ymin, jet_ymax)), style.ONLY_CLASS_LABEL_STYLE, plot_dir+'/jet_embeddings_per_jet_class')
print(" plot jet pca per event class" )
plot_PCA(jet_principle_components, np.argmax(non_zero_tiled_event_classes,axis=1), ((jet_xmin, jet_xmax),(jet_ymin, jet_ymax)), event_label_style, plot_dir+'/jet_embeddings_per_event_class')

print(" plot jet pt vs latent dims")
flattened_jet_pt = jet_features_vector.reshape(-1, jet_features_vector.shape[-1])
non_zero_flattened_jet_pt = np.squeeze(flattened_jet_pt[non_zero_jets])[:,0]

plot_latent_vs_variable(non_zero_flattened_jet_pt,non_zero_flattened_jet_embeddings, np.argmax(non_zero_flattened_jet_classes,axis=1), style.ONLY_CLASS_LABEL_STYLE, 'jet_pT',plot_dir+'/jet_embeddings_per_jet_class')
plot_latent_vs_variable(non_zero_flattened_jet_pt,non_zero_flattened_jet_embeddings, np.argmax(non_zero_tiled_event_classes,axis=1), event_label_style, 'jet_pT',plot_dir+'/jet_embeddings_per_event_class')

print(" plot jet latent per jet class" )
plot_latent(non_zero_flattened_jet_embeddings, np.argmax(non_zero_flattened_jet_classes,axis=1), style.ONLY_CLASS_LABEL_STYLE, plot_dir+'/jet_embeddings_per_jet_class')
print(" plot jet latent per event class" )
plot_latent(non_zero_flattened_jet_embeddings, np.argmax(non_zero_tiled_event_classes,axis=1), event_label_style, plot_dir+'/jet_embeddings_per_event_class')

print("===== plot jet per event class embeddings ====")

for event in event_labels.keys():
    event_type_indices = np.squeeze(np.argwhere(np.argmax(non_zero_tiled_event_classes,axis=1)==event_labels[event]))

    flattened_jet_embeddings =jet_principle_components[event_type_indices]
    print(" plot pca for " + event +" per jet class" )
    plot_PCA(flattened_jet_embeddings, np.argmax(non_zero_flattened_jet_classes[event_type_indices],axis=1), ((jet_xmin, jet_xmax),(jet_ymin, jet_ymax)), style.ONLY_CLASS_LABEL_STYLE, plot_dir+'/per_event/'+event+'/jet_embeddings_per_jet_class')
    
print("===== plot jet per jet class embeddings ====")
    
for i,jet_class in enumerate(style.ONLY_CLASS_LABEL_STYLE):

    print(" plot pca for " + jet_class +" per jet class" )
    jet_type_indices = np.squeeze(np.argwhere(np.argmax(non_zero_flattened_jet_classes,axis=1)==i))

    flattened_jet_embeddings = jet_principle_components[jet_type_indices]
    
    plot_PCA(flattened_jet_embeddings, np.argmax(non_zero_tiled_event_classes[jet_type_indices],axis=1), ((jet_xmin, jet_xmax),(jet_ymin, jet_ymax)), event_label_style, plot_dir+'/per_class/'+jet_class+'/jet_embeddings_per_event_class')
    