python test.py --problem=CVRP --model_type=MTL --checkpoint="results/20251105_133802/epoch-100.pt"

python test.py --problem=OVRPTW --model_type=MOE --num_experts=4 --routing_level=node --routing_method=input_choice --checkpoint="results/20251106_073934/epoch-100.pt"