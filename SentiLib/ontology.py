# import rdflib
from rdflib import Graph, Namespace, RDF, FOAF, Literal
from rdflib import URIRef, BNode, Literal, Namespace
from rdflib.namespace import FOAF, RDF, RDFS

def create_emonto_new(destination = '~/current_EMONTO.ttl'):
    # Generar el ttl limpio a partir del owl
    g = Graph()
    g.parse("SentiLib\models\ontologies\Emotional_Human_Ontology.owl")
    g.serialize(destination = destination)

def add_instance(name, emotion, event, intensity, object_type, emotion_category ='RobertPlutchikCategory', origin ='~/current_EMONTO.ttl', destination = '~/current_EMONTO.ttl', annotator = None, modalities = None):
    # EMONTO references
    EMONTO = Namespace('http://www.semanticweb.org/idongo/ontologies/2020/11/Emotional_Human_Ontology#')

    # Load Ontologyemo
    g = Graph()
    g.parse(origin)
    EMO = Namespace('http://www.emonto.org/') #Creating a Namespace.
    g.bind('emonto', EMONTO)
    g.bind('emo', EMO)
    g.bind('foaf', FOAF)

    person_node = EMO[name]
    # print (person_node)
    emotion_node = EMO[event + '_emotion']
    event_node = EMO[event]
    object_node = EMO[object_type]
    emotion_category_node = EMO[event + '_category']


    g.add((emotion_node, RDF.type, EMONTO['Emotion']))
    #g.add((emotion_node, EMONTO['emotionValue'], emotion_literal))
    g.add((event_node, RDF.type, EMONTO['Event']))
    g.add((event_node, RDFS.label, Literal(event)))
    g.add((person_node, RDF.type, FOAF.Person))
    g.add((person_node, RDFS.label, Literal(name)))
    g.add((object_node, RDF.type, EMONTO['Object']))
    g.add((object_node, RDFS.label, Literal(object_type)))
    g.add((emotion_category_node, RDF.type, EMONTO[emotion_category]))
    g.add((event_node, EMONTO['produces'], emotion_node))
    g.add((event_node, EMONTO['isCausedBy'], object_node))
    g.add((event_node,EMONTO['isProducedBy'], person_node))
    g.add((emotion_node,EMONTO['hasCategory'],emotion_category_node))
    g.add((emotion_category_node,EMONTO['hasIntensity'],Literal(intensity)))
    g.add((emotion_category_node,EMONTO['emotionValue'],Literal(emotion)))
    if annotator is not None:
        g.add((emotion_node,EMONTO['isAnnotatedBy'],EMONTO[annotator]))
    if modalities is not None:
        for modality in modalities:
            g.add((emotion_node,EMONTO['hasModality'],EMONTO[modality])) # agregar aqui el manejo de una lista de modalidades
    g.serialize(destination=destination)
    
def run_query(query, origin = '~/current_EMONTO.ttl'):
    g = Graph()
    g.parse(origin)
    return_value = []
    qres = g.query(query)
    for result in qres:
        return_value.append(result)
    return return_value



