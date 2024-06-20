import xml.etree.ElementTree as ET

def Anotaciones(path:str, eventos = ['Obstructive Apnea', 'Central Apnea', 'Mixed Apnea']):
    tree = ET.parse(path)
    root = tree.getroot()

    annotations = {}

    scored_events = root.find('ScoredEvents')
    for evento in eventos:
        evento_annotations = []
        for scored_event in scored_events.findall('ScoredEvent'):
            name = scored_event.find('Name').text
            if name == evento:
                start = float(scored_event.find('Start').text)
                duration = float(scored_event.find('Duration').text)
                evento_annotations.append([start, duration])
        annotations[evento] = evento_annotations

    return annotations
