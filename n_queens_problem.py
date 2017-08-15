# -*- coding: utf-8 -*-
"""
Created on Mon May 08 16:39:52 2017

@author: Michael Borden
"""
import numpy as np
#from multiprocessing import Process, Manager
import operator
import time
#import datetime
from sys import stdout
from math import sqrt

N_QUEENS          = 30        # Number of Queens MUST BE EVEN!!!
INIT_POPUL        = 1000     # N_QUEENS**2  # Size of total population
MAX_GEN           = 4000      # Maximum generations
FIT_VAL           = 0         # Minimum fitness value allowed to mate (giggidy)
MUT_RATE          = 200       # 1 / MUT_RATE
LAST_SUCC_GEN     = 0         # Last successful increase in fitness value
MAX_DEAD_GEN      = MAX_GEN/4 # Maximum generations with no progress, make drastic change
PRINT_EVERY_N_GEN = 1         # Prints information every N generatons. Costly!
BEST_FIT          = 0         # Holds the value of the current best fitness value
MUTATION_COUNT    = 0         # How man mutations have occurred
QUEEN_SYMBOL      = 'Q'       # '1\033[4mQ\033[0m'
EMPTY_SYMBOL      = '_'


# Creates an initial population of 
def create_population():
    global INIT_POPUL
    board_popul = []
    for n in range(INIT_POPUL):
        percent = (n / float(INIT_POPUL))  * 100
        stdout.write("LOADING :: %0.3f " % (percent))
        stdout.flush()
        board = []
        for m in range(N_QUEENS):
            row = np.array([EMPTY_SYMBOL] * N_QUEENS, dtype=str)
            q_pos = np.random.randint(0, N_QUEENS)
            row[q_pos] = QUEEN_SYMBOL
            board.append(row)
        board_popul.append(board)
        restart_line()
    board_popul = np.reshape(np.array(board_popul), (INIT_POPUL,N_QUEENS, N_QUEENS))
    return board_popul

# Checks the Queens in the same row
def queen_alone_in_row(board,x):
    if (np.count_nonzero(board[x] == QUEEN_SYMBOL) == 1):
            return True
    else:
        return False

# Checks the Queens in the same column
def queen_alone_in_column(board,y):
    if (np.count_nonzero(board[:,y] == QUEEN_SYMBOL) == 1):
            return True
    else:
        return False

# Checks for Queens in the same diagonals
def queen_alone_in_diagonals(board,x,y):
    global N_QUEENS
    if ((np.count_nonzero(board.diagonal(y-x) == QUEEN_SYMBOL) == 1) and (np.count_nonzero(board[:,::-1].diagonal(N_QUEENS-(x+y+1)) == QUEEN_SYMBOL)) == 1):
        return True
    else:
        return False
            
# Checks to make sure the Queen is alone in her row, column, and diagonals
def check_board(board, square):
    xp = 0
    yp = 0
    board_fit = 0   # Current boards fitness value
    for xp, row in enumerate(board):
        for yp, _ in enumerate(row):
            if (board[xp][yp] == QUEEN_SYMBOL and
                queen_alone_in_row(board,xp) and 
                queen_alone_in_column(board,yp) and 
                queen_alone_in_diagonals(board,xp,yp)):
                board_fit += 1
    if (square == True):
        board_fit = board_fit ** 2
    return board_fit

# Test the fitness of all the boards in the state space
def fitness_test(state_space, square):
    board = 0               # Current board count, by index
    fit_val = 0             # Current board's fitness
    popul_fit_list = {}     # Holds :: Key = int(index), Value = int(fitness)
    while (board < INIT_POPUL):
        fit_val = check_board(state_space[board], square)
        popul_fit_list[board] = fit_val
        board +=1 
    return popul_fit_list

# Check to see if any progress is being made, if not, maybe change the mutation rate
def progress_being_made(gen, last_succ_gen, board_popul):
    global MAX_DEAD_GEN, MUT_DEAD_RATE
    if (gen - last_succ_gen >= MAX_DEAD_GEN):
        stdout.write("\n\n\n\n################")
        stdout.write("NOT MAKING PROGRESS")
        stdout.write("#################\n\n\n\n")
        return False
    else:
        return True 

# Select mates randomly. Higher probability goes to better fitness boards
# Formula for mating probability :: Xi.fitness / sumation(Xi.fitness) 
# Which is the weight of the boards fitness relative to the sumation of populations fitness
def mate_selection(popul_fit_list):
    prob_fit_list = []          # Holds :: Key = int(board_index), Value = float(probability)
    partner_a_list = []         # Partner type A list
    partner_b_list = []         # Partner type B list
    partner_list = []           # Partners paired (A, B)
    fit_total = sum(popul_fit_list.values())
    for fit_val in popul_fit_list.values():
        if (fit_total != 0):
            prob_fit_list.extend([fit_val / float(fit_total)])
        else:
            prob_fit_list.extend([0])
    partner_a_list.extend(np.random.choice(popul_fit_list.keys(), INIT_POPUL/2, p=prob_fit_list))
    partner_b_list.extend(np.random.choice(popul_fit_list.keys(), INIT_POPUL/2, p=prob_fit_list))
    partner_list.extend(zip(partner_a_list, partner_b_list))
    return partner_list

# Mating board A with board B at a random point in the list. 
def crossover(partner_list, board_popul):
    mated_popul = []
    queen_squared = N_QUEENS**2
    for board_a, board_b in partner_list:
        split_point = np.random.randint(0, queen_squared)
        flat_board_a = np.array(board_popul[board_a]).flatten()
        flat_board_b = np.array(board_popul[board_b]).flatten()
        flat_board_b[split_point:],flat_board_a[split_point:] = flat_board_a[split_point:],flat_board_b[split_point:]
        mated_popul.append(flat_board_a.reshape(-1,N_QUEENS))
        mated_popul.append(flat_board_b.reshape(-1,N_QUEENS))
    mated_popul = np.reshape(mated_popul, (INIT_POPUL,N_QUEENS, N_QUEENS))
    return mated_popul

'''DON'T WORRY ABOUT THIS

def threaded_worker(row, mc):
    print("Working")
    for ibit, bit in enumerate(row):
        if (np.random.randint(0, MUT_RATE) == 0):
            mc += 1
            if (bit[ibit] == 1):
                bit[ibit] = 0
            else:
                bit[ibit] = 1
  
def thread_mutation(mated_list):
    buff_mate_list = []
    for board in mated_list:
        for row in board:
            manager = Manager()
            m_list = manager.list(row)
            mc_val = manager.Value('i', 0)
            multi_processing = Process(target=threaded_worker, args=(m_list))
            multi_processing.start()
            multi_processing.join()
            buff_mate_list.extend(m_list)
            MUTATION_COUNT += mc_val
    mated_list = np.reshape(buff_mate_list, (INIT_POPUL,N_QUEENS, N_QUEENS))
    return mated_list
'''

def mutation2(mated_list):
    for xp, board in enumerate(mated_list):
        for yp, bit in enumerate(board):
            pass
        

# Mutates the mated_list at a rate of 1 / mut_rate
def mutation2(mated_list, mut_rate):
    global MUTATION_COUNT, N_QUEENS, INIT_POPUL,EMPTY_SYMBOL
    for ix, board in enumerate(mated_list):
        for iy, row in enumerate(board):
            for iz, bit in enumerate(row):
                if (np.random.randint(0, mut_rate) == 2): # Why 2? Because
                    MUTATION_COUNT += 1
                    if (mated_list[ix][iy][iz] == QUEEN_SYMBOL):
                        mated_list[ix][iy][iz] = EMPTY_SYMBOL
                    else:
                        # To help even out the switches
                        if (np.random.randint(0, N_QUEENS) == 3): # Why 3? Because
                           mated_list[ix][iy][iz] = QUEEN_SYMBOL
    mated_list = np.reshape(mated_list, (INIT_POPUL,N_QUEENS, N_QUEENS))
    return mated_list

def mutation(mated_list, mut_rate):
    global MUTATION_COUNT, N_QUEENS, INIT_POPUL,EMPTY_SYMBOL
    for ix, board in enumerate(mated_list):
        for iy, row in enumerate(board):
            if (np.random.randint(0, mut_rate) == 1):
                MUTATION_COUNT += 1
                np.random.shuffle(row)
                mated_list[ix][iy] = row
    mated_list = np.reshape(mated_list, (INIT_POPUL,N_QUEENS, N_QUEENS))
    return mated_list

# This functions prints some information about the current state including the 
# current generation, current date & time, ... and also the best fitness board
def print_func(popul_fit_list, board_popul, gen, best_fit, mut_rate, square, start_time):
    global MUTATION_COUNT, PRINT_EVERY_N_GEN,N_QUEENS, INIT_POPUL
    #time_stamp = time.time()
    #date_and_time = datetime.datetime.fromtimestamp(time_stamp).strftime('%H:%M:%S')
    indx = max(popul_fit_list.iteritems(), key=operator.itemgetter(1))[0]
    bf = popul_fit_list[indx]
    seconds = time.time() - start_time
    if (square):
            bf = sqrt(bf)
    restart_line()
    if (bf > best_fit or gen % PRINT_EVERY_N_GEN == 0):
        restart_line()
        stdout.write("\n\n\n\n")
        stdout.write("Current Generation  :: %d\n" % (gen))
        stdout.write("Best Fitness        :: %d / %d\n" % (bf, N_QUEENS))
        stdout.write("Best Board's Index  :: %d / %d\n" % (indx, INIT_POPUL))
        #stdout.write("Current Time        :: %s\n" % (date_and_time))
        stdout.write("Mutation Rate       :: %0.6F%%\n" % (1 / float(mut_rate)))
        stdout.write("Mutation Count      :: %d\n" % (MUTATION_COUNT))
        stdout.write("Run Time            :: %dh:%02dm:%02ds\n" % (seconds/3600,(seconds%3600)/60,seconds%60))
        # stdout.write("Population Size     :: %d\n" % (len(board_popul)))
        x_arr = np.array(board_popul[indx]).reshape((N_QUEENS, N_QUEENS))
        stdout.write("\n".join(str('[%s]' % '|'.join(map(str, x))) for x in x_arr))
        stdout.flush()
        if (square):
            bf = bf**2
        return bf, indx
    else:
        return None, None

def restart_line():
    stdout.write('\r')
    stdout.flush()
                
'''''''''''''''''''''''''''''''''
        :::: MAIN ::::
'''''''''''''''''''''''''''''''''
def main():
    global MAX_GEN, N_QUEENS, SQUR_MUT_RATE, MUT_RATE, MUTATION_COUNT
    
    # To help view full list if too big
    if (N_QUEENS < 25):
        np.set_printoptions(threshold='nan') 
        
    start_time = time.time()
    MUTATION_COUNT = 0
    last_succ_gen = 0
    best_fit = 0        # Best fitness level
    excelsior = 0
    best_indx = 0       # Best fitness level board index
    mut_rate = N_QUEENS**2
    square = False      # Should the values be squared
    popul_fit_list = {} # Holds :: Key = int(index), Value = int(fitness)
    
    # STEP 1 :: Randomly generate state space 
    board_popul = create_population()
    print("Loading took %d seconds" % (time.time()-start_time))

    # STEP 2 :: Calculate the fitness for each board
    popul_fit_list = fitness_test(board_popul, square)

    # STEP 3 :: Count = 0
    gen = 0
    
    #and progress_being_made(gen,last_succ_gen,board_popul)
    # STEP 4 :: while gen < MAX_GEN and progress_being_made() and best_solution != N_QUEENS
    while (gen < MAX_GEN and excelsior != N_QUEENS):
        
        # STEP 5 :: Count = Count + 1
        gen += 1
        
        # STEP 6 :: Select mates from the current population
        partner_list = mate_selection(popul_fit_list)

        # STEP 7:: Apply crossover
        mated_popul = crossover(partner_list, board_popul)

        # STEP 8 :: Apply mutation
        mutated_list = mutation(mated_popul,mut_rate)
        board_popul = mutated_list
        
        # if the best fitness is over threshold, square fitness and increase mutation
        if (N_QUEENS >= 10 and best_fit >= N_QUEENS*0.85):
            square = True
            mut_rate = MUT_RATE / 4
        else:
            square = False
            mut_rate = MUT_RATE
                
        # STEP 9 :: Calculate the fitness for this new opulation of strings
        popul_fit_list = fitness_test(board_popul, square)
        
        #print("Population Fitness List ::")
        # STEP 1010 :: // end while
        #x, _ = max(zip(popul_fit_list.values(), popul_fit_list.keys()))
        bf, best_indx = print_func(popul_fit_list, board_popul, gen, best_fit, mut_rate, square, start_time)
        
        if (bf != None):
            if (last_succ_gen < bf):
                last_succ_gen = bf
                if (square):
                    excelsior = sqrt(bf)
                else:
                    excelsior = bf
            best_fit = bf
            
    # If the current board is not the best solution, print updated solution
    if (best_fit == N_QUEENS):
        np.set_printoptions(threshold='nan') # To help view full list
        stdout.write('\n')
        for x in range(N_QUEENS / 6):
            stdout.write("\t")
        stdout.write("SUCCESS!!!\n")
        x_arr = np.array(board_popul[best_indx]).reshape((N_QUEENS, N_QUEENS))
        print("\n".join(str(x) for x in x_arr))
    else:
        np.set_printoptions(threshold='nan') # To help view full list
        stdout.write('\n')
        for x in range(N_QUEENS / 6):
            stdout.write("\t")
        stdout.write("DAMN!!!\n")
        stdout.write("AGAIN!!\n")
        main()
        
if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
    
    
