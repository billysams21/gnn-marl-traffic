import json

def make_movement(dir_name, vol, is_protected, phase_id):
    return {
        "movement_dir": dir_name,
        "vol_mp": int(vol * 0.35),
        "vol_sm": int(vol * 0.65),
        "vol_ks": 0,
        "is_protected": is_protected,
        "ltor_allowed": True if dir_name == "L" else False,
        "ltor_lane_width_m": 2.5 if dir_name == "L" else 0,
        "phase_id": phase_id
    }

def make_approach(name, width, lanes, l_vol, t_vol, r_vol, l_phase, t_phase):
    return {
        "name": name,
        "width_m": width,
        "num_lanes": lanes,
        "movements": [
            make_movement("L", l_vol, True, l_phase),
            make_movement("T", t_vol, True, t_phase),
            make_movement("R", r_vol, True, t_phase)  # Right turn usually shares phase with Through
        ]
    }

def build_arterial():
    # 6000 veh/hr total in network. 
    # Arteri gets 70% ~ 4200 veh/hr, so 2100 E->W, 2100 W->E.
    # Collectors get 15% ~ 900 veh/hr (E-W).
    # N-S gets 15% ~ 900 veh/hr.
    
    intersections = []
    
    # A1, A2, A3 (Arterial E-W)
    for i_id in ["A1", "A2", "A3"]:
        intersections.append({
            "intersection_id": i_id,
            "num_phases": 2, # Phase 1: E-W, Phase 2: N-S
            "offset_s": 0,
            "approaches": [
                make_approach("Utara", 6.5, 2, 500, 2000, 500, 2, 2),
                make_approach("Selatan", 6.5, 2, 500, 2000, 500, 2, 2),
                make_approach("Timur", 16.0, 5, 1000, 10000, 1000, 1, 1),
                make_approach("Barat", 16.0, 5, 1000, 10000, 1000, 1, 1)
            ]
        })
        
    # B1, B2, B3, C1, C2, C3 (Collector E-W)
    for i_id in ["B1", "B2", "B3", "C1", "C2", "C3"]:
        intersections.append({
            "intersection_id": i_id,
            "num_phases": 2, # Phase 1: E-W, Phase 2: N-S
            "offset_s": 0,
            "approaches": [
                make_approach("Utara", 6.5, 2, 200, 1000, 200, 2, 2),
                make_approach("Selatan", 6.5, 2, 200, 1000, 200, 2, 2),
                make_approach("Timur", 6.5, 2, 200, 3000, 200, 1, 1),
                make_approach("Barat", 6.5, 2, 200, 3000, 200, 1, 1)
            ]
        })

    return {
        "settings": {
            "preferred_yellow_s": 3,
            "base_saturation_flow_per_lane": 1800,
            "clamp_cycle": True
        },
        "intersections": intersections
    }

if __name__ == "__main__":
    with open("configs/pkji_arterial_stable.json", "w") as f:
        json.dump(build_arterial(), f, indent=2)
    print("Done generating configs/pkji_arterial_stable.json")
