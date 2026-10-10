"""Generate reproducible synthetic Pune traffic complaint demonstration data."""
import csv, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLACES = ["Swargate", "Shivajinagar", "Hinjewadi", "Kothrud", "Hadapsar", "Wakad", "Baner", "Koregaon Park", "Deccan", "Pimpri", "Chinchwad", "Aundh", "Katraj", "Kharadi", "Viman Nagar", "Yerawada", "Camp", "Pune Station", "FC Road", "JM Road", "Sinhagad Road", "Warje", "Bavdhan", "Kondhwa", "Market Yard"]
TEMPLATES = {
"Congestion": ["Traffic is barely moving around {p} during the evening rush.", "A long queue of vehicles has built up near {p}; commuters are delayed.", "Vehicles are crawling by {p} and the junction is backed up.", "Please send traffic staff to {p}; congestion is stretching down the road.", "The road near {p} is packed with cars and buses this morning.", "Commuters report a severe traffic jam at {p} after the rush began."],
"Accident": ["A collision involving two cars has blocked a lane near {p}.", "There has been a road accident at {p}; please arrange assistance.", "A scooter and a car have crashed close to {p}, slowing traffic.", "Emergency help is needed after a vehicle collision by {p}.", "An overturned vehicle is causing trouble for traffic near {p}.", "Witnesses report a minor crash at {p}; one lane is obstructed."],
"Signal": ["The traffic light at {p} has been stuck on red for several cycles.", "Signals are not working at the {p} junction and traffic is confused.", "The green light never comes on near {p}; please repair the signal.", "A faulty traffic signal is causing unsafe crossings at {p}.", "The signal timer at {p} appears broken and vehicles are piling up.", "Please check the dark traffic lights near {p}; they are out of service."],
"Parking": ["Cars parked on both sides are narrowing the street at {p}.", "Illegal parking outside the shops near {p} is blocking the bus stop.", "Vehicles are left on the footpath around {p}, forcing people into the road.", "Please clear the no-parking area near {p}; traffic cannot pass easily.", "A row of parked bikes is obstructing the lane by {p}.", "Double-parked cars are blocking the entrance at {p}."],
"Construction": ["Road work near {p} has closed two lanes and created a bottleneck.", "Construction debris is scattered across the carriageway at {p}.", "A digging project by {p} has left only one lane open.", "Please add warning signs to the road repairs near {p}.", "Construction barriers are redirecting traffic onto a narrow stretch at {p}.", "Utility work around {p} has blocked the road without a clear detour."],
"Road Condition": ["A deep pothole near {p} is damaging vehicles and needs repair.", "The road surface is badly broken along {p} after the recent rain.", "There is a large crater in the road by {p}; cyclists are swerving around it.", "Loose gravel on the stretch near {p} makes riding unsafe.", "Several potholes have appeared on the road at {p}.", "The uneven pavement around {p} is a hazard for two-wheelers."]}

def main():
    rows = []
    for category, templates in TEMPLATES.items():
        for ti, template in enumerate(templates):
            for pi in range(5):
                place = PLACES[(ti * 4 + pi * 3 + list(TEMPLATES).index(category)) % len(PLACES)]
                rows.append({"complaint_text": template.format(p=place), "category": category, "template_id": f"{category}-{ti+1:02d}"})
    with open(ROOT / "data" / "sample_traffic_complaints.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["complaint_text", "category", "template_id"])
        writer.writeheader(); writer.writerows(rows)
    ner_rows = []
    for category, templates in TEMPLATES.items():
        for ti in (1, 4):
            template = templates[ti]
            for pi in range(2):
                place = PLACES[(ti * 4 + pi * 3 + list(TEMPLATES).index(category)) % len(PLACES)]
                text = template.format(p=place)
                start = text.index(place)
                ner_rows.append({"text": text, "entities": [{"start": start, "end": start + len(place), "label": "LOC"}]})
    with open(ROOT / "data" / "ner_annotated_test.jsonl", "w", encoding="utf-8") as f:
        for row in ner_rows: f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} synthetic classification rows and {len(ner_rows)} annotated NER records.")

if __name__ == "__main__": main()
