# toronto_small

Small TorontoSUMONetworks-derived external validation scenario.

Source area: Church-Wellesley neighbourhood, using arterial and collector roads.

Current network scale:

```text
22 traffic lights
114 SUMO junctions
434 SUMO edge entries
159 E1 detectors
159 E2 detectors
```

Main files:

```text
toronto_small.net.xml
toronto_small.sumocfg
toronto_small_cars.rou.xml
toronto_small_truck.rou.xml
toronto_small_cars_vtype.rou.xml
toronto_small_truck_vtype.rou.xml
e1_detectors.add.xml
e2_detectors.add.xml
```

Validated with:

```powershell
sumo -c toronto_small.sumocfg --duration-log.disable true --no-step-log true
```
