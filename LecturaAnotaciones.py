import xml.etree.ElementTree as ET

def Anotaciones(evento = 'Obstructive Apnea'):
    tree = ET.parse('homepap-lab-full-1600001-profusion.xml')
    root = tree.getroot()

    annotations = []

    scored_events = root.find('ScoredEvents')
    for scored_event in scored_events.findall('ScoredEvent'):
        name = scored_event.find('Name').text
        if (name == evento):
            start = float(scored_event.find('Start').text)
            duration = float(scored_event.find('Duration').text)
            annotations.append([start, duration])
    return annotations
