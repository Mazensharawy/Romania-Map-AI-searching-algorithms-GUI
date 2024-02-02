
from tkinter import simpledialog
import ttkbootstrap as tb
from ttkbootstrap.constants import *
import heapq
import winsound
############################
# DATA
heuristic = {
        'Arad': 366,
        'Bucharest': 0,
        'Craiova': 160,
        'Drobeta': 242,
        'Eforie': 161,
        'Fagaras': 178,
        'Giurgiu': 77,
        'Hirsova': 151,
        'Iasi': 226,
        'Lugoj': 244,
        'Mehadia': 241,
        'Neamt': 234,
        'Oradea': 380,
        'Pitesti': 98,
        'Rimnicu': 193,
        'Sibiu': 253,
        'Timisoara': 329,
        'Urziceni': 80,
        'Vaslui': 199,
        'Zerind': 374
    }
cost_graph = {
        'Arad': {'Sibiu': 140, 'Zerind': 75, 'Timisoara': 118},
        'Zerind': {'Arad': 75, 'Oradea': 71},
        'Oradea': {'Zerind': 71, 'Sibiu': 151},
        'Sibiu': {'Arad': 140, 'Oradea': 151, 'Fagaras': 99, 'Rimnicu': 80},
        'Timisoara': {'Arad': 118, 'Lugoj': 111},
        'Lugoj': {'Timisoara': 111, 'Mehadia': 70},
        'Mehadia': {'Lugoj': 70, 'Drobeta': 75},
        'Drobeta': {'Mehadia': 75, 'Craiova': 120},
        'Craiova': {'Drobeta': 120, 'Rimnicu': 146, 'Pitesti': 138},
        'Rimnicu': {'Sibiu': 80, 'Craiova': 146, 'Pitesti': 97},
        'Fagaras': {'Sibiu': 99, 'Bucharest': 211},
        'Pitesti': {'Rimnicu': 97, 'Craiova': 138, 'Bucharest': 101},
        'Bucharest': {'Fagaras': 211, 'Pitesti': 101, 'Giurgiu': 90, 'Urziceni': 8},
        'Giurgiu': {'Bucharest': 90},
        'Urziceni': {'Bucharest': 85, 'Vaslui': 142, 'Hirsova': 98},
        'Hirsova': {'Urziceni': 98, 'Eforie': 86},
        'Eforie': {'Hirsova': 86},
        'Vaslui': {'Iasi': 92, 'Urziceni': 142},
        'Iasi': {'Vaslui': 92, 'Neamt': 87},
        'Neamt': {'Iasi': 87}
    }
graph = {
        'Arad': ['Sibiu', 'Zerind', 'Timisoara'],
        'Zerind': ['Arad', 'Oradea'],
        'Oradea': ['Zerind', 'Sibiu'],
        'Sibiu': ['Arad', 'Oradea', 'Fagaras', 'Rimnicu'],
        'Timisoara': ['Arad', 'Lugoj'],
        'Lugoj': ['Timisoara', 'Mehadia'],
        'Mehadia': ['Lugoj', 'Drobeta'],
        'Drobeta': ['Mehadia', 'Craiova'],
        'Craiova': ['Drobeta', 'Rimnicu', 'Pitesti'],
        'Rimnicu': ['Sibiu', 'Craiova', 'Pitesti'],
        'Fagaras': ['Sibiu', 'Bucharest'],
        'Pitesti': ['Rimnicu', 'Craiova', 'Bucharest'],
        'Bucharest': ['Fagaras', 'Pitesti', 'Giurgiu', 'Urziceni'],
        'Giurgiu': ['Bucharest'],
        'Urziceni': ['Bucharest', 'Vaslui', 'Hirsova'],
        'Hirsova': ['Urziceni', 'Eforie'],
        'Eforie': ['Hirsova'],
        'Vaslui': ['Iasi', 'Urziceni'],
        'Iasi': ['Vaslui', 'Neamt'],
        'Neamt': ['Iasi']
    }
############################
#Breadth First Search BFS
def bfs(graph, start, goal):  #            1

    visited = [] # Explored set
    queue = [(start, [start])] # fronter

    while queue:
        current_node, path = queue.pop(0)

        if current_node == goal:
            return path  # Goal found, return the path

        visited.append(current_node)

        for neighbor in graph[current_node]:
            if neighbor not in visited :
                new_path = path + [neighbor]
                queue.append((neighbor, new_path))

    return None  # If the goal is not reached

#-------------------------=======================
#Uniform Cost Search UCS
def uniform_cost_search(cost_graph, start, goal):#            2
    # Example usage:

    priority_queue = [(0, start, [])]  # Each element is a tuple (cost, node, path)
    visited = []

    while priority_queue:
        current_cost, current_node, path_so_far = heapq.heappop(priority_queue)

        if current_node == goal:
            return current_cost, path_so_far + [current_node]  # Return cost and path

        if current_node in visited:
            continue

        visited.append(current_node)

        for neighbor, cost in cost_graph[current_node].items():
            if neighbor not in visited:
                total_cost = current_cost + cost
                heapq.heappush(priority_queue, (total_cost, neighbor, path_so_far + [current_node]))

    return None, []  # Return none if no path is found


#--------========================================
#Depth First Search DFS
def dfs_with_goal(graph, start, goal): #            1

    stack = [(start, [start])]
    visited = []

    while stack:
        current_node, path = stack.pop()

        if current_node == goal:
            return path

        if current_node in visited:
            continue

        visited.append(current_node)

        for neighbor in reversed(graph[current_node]):
            if neighbor not in visited :
              stack.append((neighbor, path + [neighbor]))

    return None
#----------===============================
#Depth Limited Search DLS
def dls_with_goal(graph, start, goal, depth_limit):#            1

    stack = [(start, [start], 0)]
    visited = []

    while stack:
        current_node, path, depth = stack.pop()
        if current_node == goal:
            return path

        if depth < depth_limit:

            if current_node in visited:
              continue

            visited.append(current_node)
            for neighbor in reversed(graph[current_node]):
              if neighbor not in visited :
                stack.append((neighbor, path + [neighbor], depth + 1))

    return
#---------=======================
#Bidirectional
def bidirectional_search(graph, start, goal):#            1

    start_queue = [(start, [start])]  # Queue for the forward search
    goal_queue = [(goal, [goal])]    # Queue for the backward search

    visited_start = []  # Set to keep track of visited nodes in the forward search
    visited_goal = []  # Set to keep track of visited nodes in the backward search

    # Initialize path lists for forward and backward search
    forward_paths = {start: [start]}
    backward_paths = {goal: [goal]}

    while start_queue and goal_queue:
        # Forward search
        current_start, path_start = start_queue.pop(0)
        visited_start.append(current_start)

        if current_start in visited_goal:
            backward_path = backward_paths[current_start]
            backward_path.pop()
            return forward_paths[current_start] + backward_path[::-1]


        for neighbor in graph[current_start]:
            if neighbor not in visited_start:
                start_queue.append((neighbor, path_start + [neighbor]))
                forward_paths[neighbor] = path_start + [neighbor]

        # Backward search
        current_goal, path_goal = goal_queue.pop(0)
        visited_goal.append(current_goal)

        if current_goal in visited_start:

          forward_path = forward_paths[current_goal]
          forward_path.pop()
          return forward_path + backward_paths[current_goal][::-1]


        for neighbor in graph[current_goal]:
            if neighbor not in visited_goal:
                goal_queue.append((neighbor, path_goal + [neighbor]))
                backward_paths[neighbor] =  path_goal + [neighbor]

    return None  # No path found
#-------==================================
#Depth Limited DFS (Iterative)
def depth_limited_dfs(graph,start, goal, depth_limit):#            1

    stack = [(start, [start], 0)]
    visited = []

    while stack:
        current_node, path, depth = stack.pop()
        if current_node == goal:
            return path

        if depth < depth_limit:

            if current_node in visited:
                continue

            visited.append(current_node)
            for neighbor in reversed(graph[current_node]):
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor], depth + 1))

    return None
def iterative_deepening_dfs(graph,start, goal):
    depth_limit = 0

    while True:
        path = depth_limited_dfs(graph ,start, goal, depth_limit)

        if path is not None:
            return path
        depth_limit += 1
    return None
#---------==========================
#Greedy Best First Search
def greedy_best_first_search(graph,start, goal,heuristic):#            1

    queue = [(heuristic[start], start, [start])]
    visited = []

    while queue:
        _, node, path = heapq.heappop(queue)

        if node == goal:
            return path  # Return the path from start to goal

        visited.append(node)

        if node in graph:
            neighbors = graph[node]
            for neighbor in neighbors:
                if neighbor not in visited:
                    new_path = path + [neighbor]
                    heapq.heappush(queue, (heuristic[neighbor], neighbor, new_path))

    return None  # No path from start to goal found
#--------==========================
#A*
def a_star_search(cost_graph,start, goal,heuristic):#            2

    queue = [(0 + heuristic[start] , start, [start])]
    visited = []

    while queue:
        cost, node, path = heapq.heappop(queue)
        cost = cost - heuristic[node]

        if node == goal:
            return cost,path  # Return the path and cost from start to goal

        visited.append(node)

        if node in cost_graph:
            neighbors = cost_graph[node]
            for neighbor, edge_cost in neighbors.items():
                if neighbor not in visited:
                    new_path = path + [neighbor]
                    new_cost = cost + edge_cost
                    total_cost = new_cost + heuristic[neighbor]
                    heapq.heappush(queue, (total_cost, neighbor, new_path))

    return None  # No path from start to goal found
#-=-=-=-=-=-=-=-=-=-=-=-=-=--

#GUI
root = tb.Window(themename="superhero")
root.title("AI Project")
width = 800 # Width 
height = 400 # Height
 
screen_width = root.winfo_screenwidth()  # Width of the screen
screen_height = root.winfo_screenheight() # Height of the screen
 
# Calculate Starting X and Y coordinates for Window
x = (screen_width/2) - (width/2)
y = (screen_height/2) - (height/2)
 
root.geometry('%dx%d+%d+%d' % (width, height, x, y))

########################################################

start_frame = tb.Frame(root) 
start_frame.pack(padx=6,pady=15,side='left', anchor='nw') 
start_label = tb.Label(start_frame,text="Start Node",font=("Arial", 10))
start_label.pack()
start_combobox = tb.Combobox(start_frame,values=['Arad', 'Zerind','Oradea','Sibiu','Timisoara', 'Lugoj', 'Mehadia', 'Drobeta', 'Craiova', 'Rimnicu', 'Fagaras', 'Pitesti', 'Bucharest', 'Giurgiu', 'Urziceni', 'Hirsova', 'Eforie', 'Vaslui', 'Iasi','Neamt'], width=15,bootstyle="success")
start_combobox.current(0)
start_combobox.pack(padx=6)
#################################################################################
goal_frame = tb.Frame(root) 
goal_frame.pack(padx=6,pady=15,side='right', anchor='ne') 
goal_label = tb.Label(goal_frame,text="Goal Node",font=("Arial", 10))
goal_label.pack()
goal_combobox = tb.Combobox(goal_frame,values=['Arad', 'Zerind','Oradea','Sibiu','Timisoara', 'Lugoj', 'Mehadia' ,'Drobeta', 'Craiova', 'Rimnicu', 'Fagaras', 'Pitesti', 'Bucharest', 'Giurgiu', 'Urziceni', 'Hirsova', 'Eforie', 'Vaslui', 'Iasi','Neamt'], width=15,bootstyle="success")
goal_combobox.current(12)
goal_combobox.pack(padx=6)
####################################################################################
algo_frame = tb.Frame(root) 

algo_frame.pack(padx=3,pady=10,side='top', anchor='n') 
algo_label = tb.Label(algo_frame,text="Algorithm",font=("Arial", 10))
algo_label.pack()
algorithm_combobox = tb.Combobox(algo_frame,values=['BFS', 'UCS', 'DFS', 'DLS', 'BS', 'IDDFS', 'GBFS', 'AStar'], width=10,bootstyle="warning")
algorithm_combobox.pack()

n_label = tb.Label(text="\n")
n_label.pack()  
###################################################################################


def find_path():
    start_city = start_combobox.get()
    goal_city = goal_combobox.get()
    algo = algorithm_combobox.get()
    cost=''
    if algo == "BFS":
            path = bfs(graph, start_city, goal_city)
            print(path)
            path = [0,path]
    elif algo == "UCS":
             path = uniform_cost_search(cost_graph, start_city, goal_city)
             cost = "Cost: "+str(path[0])
    elif algo == "DFS":
            path = dfs_with_goal(graph, start_city, goal_city)
            path = [0,path]
    elif algo == "DLS":
            depth_limit = int(simpledialog.askstring(title="DLS",prompt="Enter Depth Limit Value"))
            path = dls_with_goal(graph, start_city, goal_city, depth_limit)
            path = [0,path]
    elif algo == "BS":
            path = bidirectional_search(graph, start_city, goal_city)
            path = [0,path]
    elif algo == "IDDFS":
            path = iterative_deepening_dfs(graph, start_city, goal_city)
            path = [0,path]
    elif algo == "GBFS":
            path = greedy_best_first_search(graph, start_city, goal_city, heuristic)
            path = [0,path]
    elif algo == "AStar":
            path = a_star_search(cost_graph, start_city, goal_city, heuristic)
            cost = "Cost: "+str(path[0])
    else:
            
            result_label.config(text=f"Please Select Algorithm !",bootstyle="inverse-danger")
            winsound.Beep(1800, 500)
            #winsound.PlaySound('sradar.wav', winsound.SND_FILENAME)
    try:       
        if algo == "DLS" and path[1] == None:
            result_label.config(text=f'No path found from {start_city} to {goal_city} in limit = {depth_limit}',bootstyle="inverse-danger")
            winsound.Beep(1800, 500)
        elif path:
            result_label.config(text=f"{' => '.join(path[1])}\n{cost}",bootstyle="success")
    
        else:
            result_label.config(text="No path found.",bootstyle="inverse-danger")
            winsound.Beep(1800, 500)
    except:
        print(KeyError)

mybutton = tb.Button(root, text="Calculate", command=find_path,bootstyle="success-outline")
mybutton.pack(pady=10)


n_label = tb.Label(text="\n\n")
n_label.pack()

result_label = tb.Label(wraplength=450,text="",font=("Arial", 11))
result_label.pack()
root.mainloop()