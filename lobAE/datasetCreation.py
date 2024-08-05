import pandas as pd
import os
from tqdm import tqdm

def load_original(theMessageBookFileName, theOrderBookFileName):
    # Load and preprocess the original data
    message_org = pd.read_csv(
        filepath_or_buffer=theMessageBookFileName,
        names=['Time', 'Type', 'Order ID', 'Size', 'Price', 'Direction', 'tmp'],
        low_memory=False
    ).drop('tmp',axis=1)

    orderbook_org = pd.read_csv(
        filepath_or_buffer=theOrderBookFileName,
        names=['A1P', 'A1V', 'B1P', 'B1V', 'A2P', 'A2V', 'B2P', 'B2V', 
               'A3P', 'A3V', 'B3P', 'B3V', 'A4P', 'A4V', 'B4P', 'B4V',
               'A5P', 'A5V', 'B5P', 'B5V', 'A6P', 'A6V', 'B6P', 'B6V',
               'A7P', 'A7V', 'B7P', 'B7V', 'A8P', 'A8V', 'B8P', 'B8V',
               'A9P', 'A9V', 'B9P', 'B9V', 'A10P', 'A10V', 'B10P', 'B10V']
    ).loc[:, ['A2P', 'A2V', 'A1P', 'A1V', 'B1P', 'B1V', 'B2P', 'B2V']]

    # drop duplicates (Some changes happen outside the top 2 levels)
    orderbook_org_shifted = orderbook_org.shift()
    orderbook_org_unique = orderbook_org[orderbook_org.ne(orderbook_org_shifted).any(axis=1)]
    orderbook_org = orderbook_org_unique
    message_org = message_org.loc[orderbook_org.index]  #Sample the messagebook accordingly 

    # Convert 'Time' column to nanoseconds timestamp
    message_org['Time'] = message_org['Time'] * 1e9
    message_org['Time'] = pd.to_datetime(message_org['Time'], unit='ns')
    # Set 'Time' as index for both dataframes
    message_org.set_index('Time', inplace=True)
    orderbook_org['Time'] = message_org.index
    orderbook_org.set_index('Time', inplace=True)
    data = pd.concat([message_org,orderbook_org],axis=1)
    data['Seq'] = range(0,len(data))
    orderbook_org['Seq'] = range(0,len(orderbook_org))
    message_org['Seq'] = range(0,len(message_org))
    return orderbook_org, message_org, data

def orderbook_resampling(orderbook_org):
    df = orderbook_org.copy()
    # Resample orderbook to 1s intervals by taking the last state in each interval
    orderbook_resampled_1s = df.resample('1s').last().dropna()['Seq']
    # Combine the indices of mid-price changes and 1s intervals
    df.set_index('Seq',inplace=True)
    orderbook_resampled = df.loc[orderbook_resampled_1s,:]
    return orderbook_resampled

def get_act_positions_cash_invt_cancel(orderbook_org, message_org, orderbook_resampled):
    def cancel_positions_beyond_top2(lob_state, agent_position):
        # Function to check and cancel positions beyond the top 2 levels
        level2_ask_price = lob_state['A2P']
        level2_bid_price = lob_state['B2P']
        new_agent_position = agent_position.copy()
        for order_id, (price, _) in agent_position.items():
            if (price < level2_bid_price or price > level2_ask_price):
                del new_agent_position[order_id]
        return new_agent_position
    
    orderbook = orderbook_org.set_index('Seq')
    messagebook = message_org.set_index('Seq')

    # Track the agent's positions, cash and inventory
    positions = []
    cash = []
    inventory = []
    # Initialize agent's position, cash, and inventory
    agent_position = {}
    agent_cash = 1e9
    agent_inventory = 1e6

    cancel_list = []  # record all the cancels by our agent between 2 observed states
    actions_list = []  # record all the actions by our agent

    # Iterate through each resampled time point
    for t,i in zip(orderbook_resampled.head(-1).index, range(len(orderbook_resampled)-1)):
        positions.append(agent_position)
        agent_position = agent_position.copy()  # make a copy to prevent changing the oringinal
        cash.append(agent_cash)
        inventory.append(agent_inventory)
        
        # Get the next order in the messagebook right after L(t_n)
        next_order = messagebook.loc[t+1]
        seq = next_order.name
        order_id = next_order['Order ID']
        order_type = next_order['Type']
        price = next_order['Price']
        size = next_order['Size']
        direction = next_order['Direction']
        
        if order_type == 1:
            # Agent posts a limit order
            agent_position[order_id] = (price, size)
            new_action = pd.DataFrame({
                'Seq': seq,
                'Action': 'Limit Order',
                'Order ID': order_id,
                'Price': price,
                'Volume': size,
                'Type': order_type,
                'Direction': direction
            }, index=[0]
            )
            actions_list.append(new_action)
        elif order_type in [2, 3]:
            # Cancel Order
            if order_id in agent_position:
                current_price, current_volume = agent_position[order_id]
                new_volume = current_volume - size
                if new_volume > 0:
                    agent_position[order_id] = (current_price, new_volume)
                else:
                    del agent_position[order_id]
                new_action = pd.DataFrame({
                    'Seq': seq,
                    'Action': 'Cancel Order',
                    'Order ID': order_id,
                    'Price': price,
                    'Volume': size,
                    'Type': order_type,
                    'Direction': direction
                }, index=[0]
                )
                actions_list.append(new_action)
            else:
                new_action = pd.DataFrame({
                    'Seq': seq,
                    'Action': 'No action',
                    'Order ID': None,
                    'Price': None,
                    'Volume': None,
                    'Type': None,
                    'Direction': None
                }, index=[0]
                )
                actions_list.append(new_action)
        elif order_type == 4:
            # Market Order
            if order_id in agent_position:
                # Market order by other participants and it hits our agent's position
                hit_orders = [oid for oid, (p, v) in agent_position.items() if p == price]
                oid = hit_orders[0]
                current_price, current_volume = agent_position[oid]
                new_volume = current_volume - size
                agent_cash -= size * price * direction / 10000  # Update cash (price scaled by 10000)
                agent_inventory += size * direction  # Update inventory
                if new_volume > 0:
                    agent_position[oid] = (current_price, new_volume)
                else:
                    del agent_position[oid]
                new_action = pd.DataFrame({
                    'Seq': seq,
                    'Action': 'No action',
                    'Order ID': None,
                    'Price': None,
                    'Volume': None,
                    'Type': None,
                    'Direction': None
                }, index=[0]
                )
                actions_list.append(new_action)
            else:
                # Market order by our agent
                agent_cash += size * price * direction / 10000  # Update cash (price scaled by 10000)
                agent_inventory -= size * direction  # Update inventory
                new_action = pd.DataFrame({
                    'Seq': seq,
                    'Action': 'Market Order',
                    'Order ID': order_id,
                    'Price': price,
                    'Volume': size,
                    'Type': order_type,
                    'Direction': direction
                }, index=[0]
                )
                actions_list.append(new_action)
            
        # Check and cancel positions beyond the top 2 levels
        agent_position = cancel_positions_beyond_top2(orderbook.loc[seq], agent_position)

        # Check other orders in messagebook that may affect the agent's position, cash and inventory
        other_orders = messagebook[(messagebook.index > t+1) & (messagebook.index <= orderbook_resampled.iloc[i+1].name)]
        for seq, order in other_orders.iterrows():
            order_id = order['Order ID']
            order_type = order['Type']
            price = order['Price']
            size = order['Size']
            direction = order['Direction']

            if order_type == 1:
                # Limit order by other participants, do nothing
                pass
            elif order_type in [2, 3]:
                # Cancel Order
                if order_id in agent_position:  # cancels by our agent
                    # add the last lob state ro the resampled orderbook
                    cancel_list.append(orderbook.loc[seq-1])
                    positions.append(agent_position)
                    agent_position = agent_position.copy()  # make a copy to prevent changing the oringinal
                    cash.append(agent_cash)
                    inventory.append(agent_inventory)
                    current_price, current_volume = agent_position[order_id]
                    new_volume = current_volume - size
                    if new_volume > 0:
                        agent_position[order_id] = (current_price, new_volume)
                    else:
                        del agent_position[order_id]
                    new_action = pd.DataFrame({
                        'Seq': seq,
                        'Action': 'Cancel Order',
                        'Order ID': order_id,
                        'Price': price,
                        'Volume': size,
                        'Type': order_type,
                        'Direction': direction
                    }, index=[0]
                    )
                    actions_list.append(new_action)
            elif order_type == 4:
                # Market Order
                if order_id in agent_position:
                    # Market order by other participants and it hits our agent's position
                    hit_orders = [oid for oid, (p, v) in agent_position.items() if p == price]
                    oid = hit_orders[0]
                    current_price, current_volume = agent_position[oid]
                    new_volume = current_volume - size
                    agent_cash -= size * price * direction / 10000  # Update cash (price scaled by 10000)
                    agent_inventory += size * direction  # Update inventory
                    if new_volume > 0:
                        agent_position[oid] = (current_price, new_volume)
                    else:
                        del agent_position[oid]
            
            # Check and cancel positions beyond the top 2 levels
            agent_position = cancel_positions_beyond_top2(orderbook.loc[seq], agent_position)
    actions = pd.concat(actions_list,ignore_index=True)
    return actions, positions, cash, inventory, cancel_list

def data_merge(orderbook_resampled,positions,cash,inventory,actions,cancel_list):
    cancels = pd.concat(cancel_list)
    cancels = pd.concat(cancel_list,axis=1).T
    orderbook_resampled = pd.concat([orderbook_resampled, cancels]).sort_index()
    data = pd.concat([
        orderbook_resampled.reset_index(),
        pd.Series(cash, name='Cash'),
        pd.Series(inventory, name='Inventory'),
        pd.Series(positions, name='Position'),
        actions.drop('Seq', axis=1)
    ],axis=1)
    return data

def main():
    sDate = '20190101'
    eDate = '20200930'
    dataPath = '/Users/sw/Working Space/Python/nmdp_rl/data/AAPL_2019-01-01_2020-09-27_10/'
    storePath = '/Users/sw/Working Space/Python/nmdp_rl/data/dataset/'
    for d in tqdm(pd.date_range(sDate, eDate)): # TODO: business days try catch
        theMessageBookFileName = dataPath + "AAPL_" + d.strftime("%Y-%m-%d") + "_34200000_57600000_message_10.csv"
        theOrderBookFileName = dataPath + "AAPL_" + d.strftime("%Y-%m-%d") + "_34200000_57600000_orderbook_10.csv"
        if ("AAPL_" + d.strftime("%Y-%m-%d") + "_34200000_57600000_message_10.csv") not in os.listdir(dataPath): continue
        orderbook_org, message_org, _ = load_original(theMessageBookFileName, theOrderBookFileName)
        orderbook_resampled = orderbook_resampling(orderbook_org)
        actions, positions, cash, inventory, cancel_list = get_act_positions_cash_invt_cancel(orderbook_org, message_org, orderbook_resampled)
        data = data_merge(orderbook_resampled,positions,cash,inventory,actions,cancel_list)
        data.to_csv(storePath + 'AAPL_' + d.strftime("%Y-%m-%d") + '_dataset_2ls.csv')

if __name__ == "__main__":
    main()

