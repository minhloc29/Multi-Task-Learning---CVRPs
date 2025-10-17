from dataclasses import dataclass
import torch
from VRProblemDef import get_random_problems_mixed, augment_xy_data_by_8_fold

# Giả định VRProblemDef có hàm get_random_problems_mixed và augment_xy_data_by_8_fold

@dataclass
class Reset_State:
    depot_xy: torch.Tensor = None
    # shape: (batch, 1, 2)
    node_xy: torch.Tensor = None
    # shape: (batch, problem, 2)
    node_demand: torch.Tensor = None
    # shape: (batch, problem)
    node_earlyTW: torch.Tensor = None
    # shape: (batch, problem)
    node_lateTW: torch.Tensor = None
    # shape: (batch, problem)


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor = None
    POMO_IDX: torch.Tensor = None
    # shape: (batch, pomo)
    selected_count: int = None
    load: torch.Tensor = None
    # shape: (batch, pomo)
    time: torch.Tensor = None
    # shape: (batch, pomo)
    route_open: torch.Tensor = None
    # shape: (batch, pomo)
    length: torch.Tensor = None
    # shape: (batch, pomo)
    
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, problem+1)
    finished: torch.Tensor = None
    # shape: (batch, pomo)


class VRPEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.env_params = env_params
        self.problem_size = env_params['problem_size']
        self.pomo_size = env_params['pomo_size']
        self.problem_type = env_params['problem_type']

        self.problem_type_list = env_params.get('problem_type_list', ['CVRP', 'VRPTW', 'OVRP', 'VRPB', 'VRPL'])
        
        # Nếu problem_type không có trong list, thêm vào (dành cho mode single-task)
        #if self.problem_type not in self.problem_type_list:
             # Tìm vị trí an toàn, ví dụ: vị trí cuối cùng.
            #if len(self.problem_type_list) < 5 or self.problem_type not in self.problem_type_list:
                #self.problem_type_list = ['CVRP', 'VRPTW', 'OVRP', 'VRPB', 'VRPL'] # Reset list chuẩn
            #if self.problem_type not in self.problem_type_list:
                #self.problem_type_list.append(self.problem_type)


        self.FLAG__use_saved_problems = False
        self.saved_depot_xy = None
        self.saved_node_xy = None
        self.saved_node_demand = None
        self.saved_index = None

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, pomo)
        self.depot_node_xy = None
        # shape: (batch, problem+1, 2)
        self.depot_node_demand = None
        # shape: (batch, problem+1)
        self.depot_node_earlyTW = None
        # shape: (batch, problem+1)
        self.depot_node_lateTW = None
        # shape: (batch, problem+1)
        self.depot_node_servicetime = None
        # shape: (batch, problem+1)
        self.length = None
        # shape: (batch, pomo)
        
        # ⚠️ BỎ CÁC CỜ TOÀN CỤC: Chúng ta sẽ tính toán chúng dựa trên task_label trong hàm step
        # Tạm thời vẫn giữ nguyên nhưng không dùng để điều khiển logic
        self.attribute_c = False
        self.attribute_tw = False
        self.attribute_o = False
        self.attribute_b = False 
        self.attribute_l = False


        # Dynamic-1
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~)

        # Dynamic-2
        ####################################
        self.at_the_depot = None
        # shape: (batch, pomo)
        self.load = None
        # shape: (batch, pomo)
        self.time = None
        # shape: (batch, pomo)
        self.route_open= None
        # shape: (batch, pomo)
        self.length= None
        # shape: (batch, pomo)
        self.visited_ninf_flag = None
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = None
        # shape: (batch, pomo, problem+1)
        self.finished = None
        # shape: (batch, pomo)

        # states to return
        ####################################
        self.reset_state = Reset_State()
        self.step_state = Step_State()
        self.task_label = None  # <--- NEW

    def _get_task_label(self):
        """Convert problem type string to numeric label (không dùng trong mode mixed-task)."""
        mapping = {
             # Giả định mapping cho 5 task đầu tiên:
            "CVRP": 0,
            "VRPTW": 1,
            "OVRP": 2,
            "VRPB": 3, 
            "VRPL": 4 
            # Bạn cần đảm bảo các problem type trong self.problem_type_list
            # tương ứng với các chỉ số 0, 1, 2, 3, 4, ...
        }
        # Chỉ trả về label của problem_type hiện tại (mode single-task)
        return self.problem_type_list.index(self.problem_type)
    
    def use_saved_problems(self, filename, device):
        # ... (Không thay đổi) ...
        self.FLAG__use_saved_problems = True

        loaded_dict = torch.load(filename, map_location=device)
        self.saved_depot_xy = loaded_dict['depot_xy']
        self.saved_node_xy = loaded_dict['node_xy']
        self.saved_node_demand = loaded_dict['node_demand']
        self.saved_node_earlyTW = loaded_dict['node_earlyTW']
        self.saved_node_lateTW = loaded_dict['node_lateTW']
        self.saved_node_servicetime = loaded_dict['node_serviceTime']
        self.saved_route_open = loaded_dict['route_open']
        self.saved_route_length = loaded_dict['route_length_limit']
        self.saved_index = 0

    def load_problems(self, batch_size, aug_factor=1):
        self.batch_size = batch_size

        # ----- Multi-task mixing -----
        # Sử dụng len(self.problem_type_list) để đảm bảo khớp với số lượng task
        num_tasks = len(self.problem_type_list) 
        labels = torch.randint(low=0, high=num_tasks, size=(batch_size,), dtype=torch.long)
        self.task_label = labels  # <--- mỗi phần tử trong batch giờ là 1 task khác nhau

        depot_xy = torch.zeros(batch_size, 1, 2)
        node_xy = torch.zeros(batch_size, self.problem_size, 2)
        node_demand = torch.zeros(batch_size, self.problem_size)
        node_earlyTW = torch.zeros(batch_size, self.problem_size)
        node_lateTW = torch.zeros(batch_size, self.problem_size)
        node_servicetime = torch.zeros(batch_size, self.problem_size)
        route_open = torch.zeros(batch_size, self.problem_size)
        route_length_limit = torch.zeros(batch_size, self.pomo_size)

        # Sinh dữ liệu cho từng task riêng rồi ghép lại
        for t in range(num_tasks):
            # 💡 Chú ý: Cần import get_random_problems_mixed để đoạn này hoạt động
            idx = (labels == t).nonzero(as_tuple=True)[0]
            if len(idx) == 0:
                continue
            depot_xy_t, node_xy_t, node_demand_t, node_earlyTW_t, node_lateTW_t, node_servicetime_t, route_open_t, route_length_limit_t = \
                get_random_problems_mixed(len(idx), self.problem_size, self.problem_type_list[t])
            
            depot_xy[idx] = depot_xy_t
            node_xy[idx] = node_xy_t
            node_demand[idx] = node_demand_t
            node_earlyTW[idx] = node_earlyTW_t
            node_lateTW[idx] = node_lateTW_t
            node_servicetime[idx] = node_servicetime_t
            route_open[idx] = route_open_t
            route_length_limit[idx] = route_length_limit_t

        # ----- Augmentation -----
        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                # 💡 Chú ý: Cần import augment_xy_data_by_8_fold để đoạn này hoạt động
                depot_xy = augment_xy_data_by_8_fold(depot_xy)
                node_xy = augment_xy_data_by_8_fold(node_xy)
                node_demand = node_demand.repeat(8, 1)
                node_earlyTW = node_earlyTW.repeat(8, 1)
                node_lateTW = node_lateTW.repeat(8, 1)
                node_servicetime = node_servicetime.repeat(8, 1)
                route_open = route_open.repeat(8, 1)
                route_length_limit = route_length_limit.repeat(8, 1)
                self.task_label = self.task_label.repeat(8) # task_label phải được lặp lại
            else:
                raise NotImplementedError

        # ----- Gộp depot + node -----
        self.route_open = route_open
        self.length = route_length_limit

        self.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        depot_demand = torch.zeros(size=(self.batch_size, 1))
        self.depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)

        depot_earlyTW = torch.zeros(size=(self.batch_size, 1))
        depot_lateTW = 4.6 * torch.ones(size=(self.batch_size, 1))
        depot_servicetime = torch.zeros(size=(self.batch_size, 1))
        self.depot_node_earlyTW = torch.cat((depot_earlyTW, node_earlyTW), dim=1)
        self.depot_node_lateTW = torch.cat((depot_lateTW, node_lateTW), dim=1)
        self.depot_node_servicetime = torch.cat((depot_servicetime, node_servicetime), dim=1)

        # ----- Index setup -----
        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

        self.reset_state.depot_xy = depot_xy
        self.reset_state.node_xy = node_xy
        self.reset_state.node_demand = node_demand
        self.reset_state.node_earlyTW = node_earlyTW
        self.reset_state.node_lateTW = node_lateTW

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX

        # ⚠️ BỎ QUA CÁC CỜ TOÀN CỤC Ở ĐÂY - CHỈ SỬ DỤNG task_label trong hàm step
        self.attribute_c = (node_demand.sum() > 0)
        self.attribute_tw = (node_lateTW.sum() > 0)
        self.attribute_o = (route_open.sum() > 0)
        self.attribute_l = (route_length_limit.sum() > 0)


    def reset(self):
        # ... (Giữ nguyên) ...
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~)

        self.at_the_depot = torch.ones(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)
        self.load = torch.ones(size=(self.batch_size, self.pomo_size))
        # shape: (batch, pomo)
        self.time = torch.zeros(size=(self.batch_size, self.pomo_size))
        # shape: (batch, pomo)
        self.length = 3.0*torch.ones(size=(self.batch_size, self.pomo_size))
        # # shape: (batch, pomo)
        self.route_open = torch.zeros((self.batch_size, self.pomo_size))
        # shape: (batch, pomo)
        self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size+1))
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size+1))
        # shape: (batch, pomo, problem+1)
        self.finished = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)

        reward = None
        done = False
        return self.reset_state, self.task_label, done

    def pre_step(self):
        # ... (Giữ nguyên) ...
        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished
        self.step_state.time = self.time
        self.step_state.route_open = self.route_open
        self.step_state.length = self.length.clone()

        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        # selected.shape: (batch, pomo)

        # Dynamic-1
        ####################################
        self.selected_count += 1
        self.current_node = selected
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~)

        # Dynamic-2
        ####################################

        self.at_the_depot = (selected == 0)

        #### update load information ###

        demand_list = self.depot_node_demand[:, None, :].expand(-1, self.pomo_size, -1)
        # shape: (batch, pomo, problem+1)
        gathering_index = selected[:, :, None]
        # shape: (batch, pomo, 1)
        selected_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
        # shape: (batch, pomo)

        self.load -= selected_demand
        self.load[self.at_the_depot] = 1 # refill loaded at the depot


        #### mask nodes if load exceed (CVRP, VRPTW, VRPL) ###

        self.visited_ninf_flag[self.BATCH_IDX, self.POMO_IDX, selected] = float('-inf')
        # shape: (batch, pomo, problem+1)
        self.visited_ninf_flag[:, :, 0][~self.at_the_depot] = 0  # depot is considered unvisited, unless you are AT the depot

        self.ninf_mask = self.visited_ninf_flag.clone()
        round_error_epsilon = 0.000001
        demand_too_large = self.load[:, :, None] + round_error_epsilon < demand_list
        # shape: (batch, pomo, problem+1)
        
        # 💡 Sửa: Chỉ áp dụng Demand-mask cho các task có Demand (CVRP, VRPTW, VRPL, ...)
        # Giả định CVRP=0, VRPTW=1, VRPL=4. OVRP=2, VRPB=3 có thể không dùng demand-mask 
        # Cần xác định task nào có capacity. Ở đây ta dùng CVRPTW và CVRP/VRPL là có capacity
        
        # Tạo mask boolean cho các mẫu có capacity (C-tasks)
        is_C_task = ((self.task_label == 0) | (self.task_label == 1) | (self.task_label == 4))[:, None, None].expand_as(self.ninf_mask)
        
        # Chỉ áp dụng demand_too_large cho các mẫu có capacity
        demand_mask_to_apply = demand_too_large & is_C_task
        self.ninf_mask[demand_mask_to_apply] = float('-inf')
        # shape: (batch, pomo, problem+1)
        
        
        #### update time&distance information ###

        servicetime_list = self.depot_node_servicetime[:, None, :].expand(-1, self.pomo_size, -1)
        gathering_index = selected[:, :, None]
        selected_servicetime = servicetime_list.gather(dim=2,index=gathering_index).squeeze(dim=2)
        
        earlyTW_list = self.depot_node_earlyTW[:, None, :].expand(-1, self.pomo_size, -1)
        selected_earlyTW = earlyTW_list.gather(dim=2,index=gathering_index).squeeze(dim=2)

        xy_list = self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1,-1)
        gathering_index = selected[:, :, None,None].expand(-1,-1,-1,2)
        selected_xy = xy_list.gather(dim=2, index=gathering_index).squeeze(dim=2)

        if self.selected_node_list.size()[2] == 1:
            gathering_index_last = self.selected_node_list[:, :, -1][:,:,None,None].expand(-1,-1,-1,2)
        else:
            gathering_index_last = self.selected_node_list[:, :, -2][:,:,None,None].expand(-1,-1,-1,2)
        
        last_xy = xy_list.gather(dim=2, index=gathering_index_last).squeeze(dim=2)
        selected_time = ((selected_xy - last_xy)**2).sum(dim=2).sqrt()

        # ------------------------------------------------------------------------------------------------------
        # 💡 SỬA: Logic Time Window (VRPTW) - Dùng task_label=1
        # ------------------------------------------------------------------------------------------------------
        
        # Tạo mask boolean cho các mẫu là VRPTW (Giả định VRPTW là task_label == 1)
        is_tw_sample = (self.task_label == 1)[:, None].expand(-1, self.pomo_size) # shape: (batch, pomo)

        # 1. Cập nhật self.time (Chỉ cho các mẫu VRPTW)
        time_update = torch.max((self.time + selected_time), selected_earlyTW)
        time_update += selected_servicetime
        time_update[self.at_the_depot] = 0 # refill time at the depot
        
        # Chỉ cập nhật self.time cho các mẫu VRPTW
        self.time = torch.where(is_tw_sample, time_update, self.time)

        # 2. Logic Masking (Chỉ áp dụng cho VRPTW)
        if (self.attribute_tw): # Vẫn dùng cờ này để kiểm tra xem TW có được tải hay không (nhưng không dùng để điều khiển logic)
            # Tạo mask mở rộng cho VRPTW
            is_tw_mask_expanded = is_tw_sample[:, :, None].expand_as(self.ninf_mask) # shape: (batch, pomo, problem+1)

            time_to_next = ((selected_xy[:,:,None,:].expand(-1,-1,self.problem_size+1,-1) - xy_list)**2).sum(dim=3).sqrt()
            time_too_late = self.time[:, :, None] + time_to_next > self.depot_node_lateTW[:, None, :].expand(-1, self.pomo_size, -1)
            
            # unmask the the zero late TW 
            time_too_late[self.depot_node_lateTW[:, None, :].expand(-1, self.pomo_size, -1) == 0]= 0 
            
            # Áp dụng ninf_mask chỉ cho các mẫu VRPTW đang quá giờ
            mask_to_apply = time_too_late & is_tw_mask_expanded
            self.ninf_mask[mask_to_apply] = float('-inf')

        # ------------------------------------------------------------------------------------------------------
        # 💡 SỬA: Logic Route Length (VRPL) - Dùng task_label=4
        # ------------------------------------------------------------------------------------------------------
        
        # Tạo mask boolean cho các mẫu là VRPL (Giả định VRPL là task_label == 4)
        is_l_sample = (self.task_label == 4)[:, None].expand(-1, self.pomo_size) # shape: (batch, pomo)

        # update route duration (length) attribute if it is used
        if (self.attribute_l): # Vẫn dùng cờ này để kiểm tra xem Length có được tải hay không
            # 1. Cập nhật self.step_state.length (Chỉ cho các mẫu VRPL)
            
            # Tính toán giá trị length mới
            length_update = self.step_state.length.clone() # Dùng clone() vì nó là Step_State
            length_update -= selected_time
            length_update[self.at_the_depot] = self.length[0][0] # refill length at the depot

            # Chỉ cập nhật length cho các mẫu VRPL
            self.step_state.length = torch.where(is_l_sample, length_update, self.step_state.length)
            
            # 2. Logic Masking (Chỉ áp dụng cho VRPL)
            is_l_mask_expanded = is_l_sample[:, :, None].expand_as(self.ninf_mask) # shape: (batch, pomo, problem+1)

            length_to_next = ((selected_xy[:,:,None,:].expand(-1,-1,self.problem_size+1,-1) - xy_list)**2).sum(dim=3).sqrt()
            depot_xy = xy_list[:,:,0,:]
            next_to_depot =  ((depot_xy[:,:,None,:].expand(-1,-1,self.problem_size+1,-1)  - xy_list)**2).sum(dim=3).sqrt()
            
            # if open attribute is used, the distance return to depot is not counted
            if self.attribute_o:
                # 💡 Giả định Open Route (OVRP) là task_label = 2
                is_o_sample = (self.task_label == 2)[:, None].expand(-1, self.pomo_size)
                # Dùng is_o_sample để điều khiển route_open 
                
                # Check cho cả VRPL (task=4) và OVRP (task=2)
                # Tuy nhiên, logic này có vẻ bị lẫn lộn giữa VRPL và OVRP. 
                # Giả định bạn chỉ muốn áp dụng Length Limit cho VRPL (task=4)
                
                # Length check: chỉ tính length_to_next (không về depot) nếu là VRPL (task=4) và là OVRP (task=2)
                
                # Logic cũ (có thể sai nếu task=2 không phải là Length Limited)
                # length_too_small = self.step_state.length[:, :, None] - round_error_epsilon < length_to_next 

                # Logic mới: Nếu là OVRP (task=2) hoặc VRPL (task=4) mà có route_open=1, thì không tính về depot
                # Dùng is_l_sample để lọc ra các mẫu cần kiểm tra Length
                
                # Ràng buộc length cho VRPL, không tính về depot nếu route_open == 1
                length_too_small_L = (self.step_state.length[:, :, None] - round_error_epsilon < length_to_next)
                
            else: # (Không phải Open Route)
                # Áp dụng cho VRPL (task=4) và CVRP (task=0) nếu chúng có length limit
                length_too_small_L = (self.step_state.length[:, :, None] - round_error_epsilon < (length_to_next + next_to_depot ))
            
            # 💡 Chỉ áp dụng Length-mask cho các mẫu là VRPL
            mask_to_apply = length_too_small_L & is_l_mask_expanded
            self.ninf_mask[mask_to_apply] = float('-inf')

            # is_l_sample có shape (batch, pomo), nên không cần chỉ số thứ 3 [:,:,0]
            # Bạn chỉ cần dùng is_l_sample.
            mask_condition = is_l_sample & (~self.at_the_depot) 

            # Áp dụng điều kiện mask (batch, pomo) lên chỉ mục node 0 của ninf_mask (batch, pomo, 1)
            self.ninf_mask[:, :, 0][mask_condition] = 0
        
            
            
        newly_finished = (self.visited_ninf_flag == float('-inf')).all(dim=2)
        # shape: (batch, pomo)
        self.finished = self.finished + newly_finished
        # shape: (batch, pomo)

        # do not mask depot for finished episode.
        self.ninf_mask[:, :, 0][self.finished] = 0

        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

        # returning values
        done = self.finished.all()
        if done:
            reward = -self._get_travel_distance()  # note the minus sign!
        else:
            reward = None

        return self.step_state, reward, done

    def _get_travel_distance(self):
        # ... (Giữ nguyên) ...
        gathering_index = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        all_xy = self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)

        ordered_seq = all_xy.gather(dim=2, index=gathering_index)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq-rolled_seq)**2).sum(3).sqrt()

        # 💡 SỬA: Chỉ đặt lại về 0 cho các mẫu OVRP (task_label = 2)
        is_o_sample = (self.task_label == 2)[:, None].expand(-1, self.pomo_size)
        is_o_expanded = is_o_sample[:, :, None].expand_as(segment_lengths)
        
        # if the target problem is open route the distance return to depot will be set as 0       
        # Chỉ áp dụng mask cho các mẫu OVRP (và nó phải là cuối route)
        is_last_node_depot = (self.selected_node_list.roll(dims=2, shifts=-1)==0)

        # Chỉ áp dụng setting to zero nếu: 1. Là mẫu OVRP (is_o_expanded) AND 2. Là bước cuối route (is_last_node_depot)
        segment_lengths = torch.where(is_o_expanded & is_last_node_depot, torch.zeros_like(segment_lengths), segment_lengths)

        travel_distances = segment_lengths.sum(2)
        return travel_distances

    def get_node_seq(self):
        # ... (Giữ nguyên) ...
        gathering_index = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        all_xy = self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)

        ordered_seq = all_xy.gather(dim=2, index=gathering_index)
        return gathering_index,ordered_seq
