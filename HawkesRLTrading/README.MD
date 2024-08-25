HawkesRLTrading is an entity-based discrete event reinforcement-learning simulation environment for equities trading. Currently, it only supports one independent reinforcement learning agent, alongside multiple other non-intelligent random noise agents that can be configured. Further support will be implemented for multi-agent trading. 

Architecture of HawkesRLTrading:
The RL module has been designed to be extensible, general and to minimize code duplication. All our gym environments inherit from a class TradingEnvironment and use the same following shared "step" and “update” functions. The step function is not specific to the trading problem. Similarly, the TradingEnv also contains an ArrivalModels class which universally inherits a "getnext_arrival" method that can be implemented for various different types of order arrival modelling. Currently, only a Hawkes Model has been implemented via this architecture. 

The core of the simulation is a discrete event-based kernel that supports flow of information between classes of objects known as Entities.

An entity is a relevant party that has to exchange information during the simulation. Entities are split into trading agent entities and exchange entities. Trading agent entities refer to the various GYM and non-gym trading agents, and the exchange entity is responsible for the exchange of information on behalf of the market exchange itself. 

The kernel resides in the Kernel class in the default package and is required in all simulations. All messages must pass through the kernel’s event queue. It is also required for the upkeep and coordination of the simulation itself. Key features of the Kernel include:
    1. Global Virtual time: GVT is the latest simulated time for which all messages are guaranteed to have been processed. The kernel tracks GVT as the simulation progresses. It is usually the case that GVT advances much more quickly than wall clock time, but for very complex scenarios, it may not. This value is not available to the agents
    2. Current time per agent: The kernel tracks a “current time” for participating agent which is incremented upon any return from Agent.set_wakeup(). In situations where the current time for the agent is “in the future” (i.e., larger than GVT), the kernel will delay delivery of messages or wakeup calls to this agent until GVT catches up.
    3. Deterministic but Random execution: The kernel accepts a single pseudo-random number generator which is then used to generate seeds for various stochastic processes. In this way, changes in the behavior of other agents will be caused by a changed simulation environment and not due to stochastic perturbation.

All Simulation Entities are inherited from the Entity abstract base class. All participants in a simulation, whether a trading agent or the Exchange module must inherit from this Entity class. This Entity class implements a number of necessary methods that allow basic participation in the simulation with the kernel. The following methods must be supported by all Entity agents:
    1. sendMessage(senderID, recipientID, message)
    2. sendBatchMessage(senderID, recipientIDs, message)
    3. receiveMessage(senderID, message)
    3. Kernel_start()
    4. Kernel_terminate()
The TradingAgent and the Exchange classes inherit separately from this Entity base class. In addition, the TradingAgent classes and its children subclasses must all support the WakeUp() and set_WakeUp(time) methods, which respectively notify a TradingAgent it is their turn to trade and sets a definite time in the future when they can perform an action again. Agents also have the option to be set up with the parameter on_trade=True, which additionally includes the agent in the set of entities that are notified whenever a trade has taken place. This then gives them the option to perform an action upon a trade in the Exchange.

The Exchange class is responsible for the implementation of a limit order book for one specific symbol which is default to the name of "XYZ". An order book tracks all open orders for a single stock symbol. All order book activity is logged through the exchange entity. The Exchange Class implements the following functionalities:

• Order Matching Attempts to match the incoming order against the appropriate side of the order book. The best price match is selected. In the case of multiple orders at the same price, the oldest order is selected. This occurs either when a market order is made, or when the bid-ask prices co-incide at the same pricepoint.

• Partial Execution Either the incoming market order or the matched limit order may be partially executed. When the
matched limit order is partially executed, the order is left in the book with its quantity reduced. It also sends a notification to a TradingAgent if the partially executed limit order belonged to them. Trading agents receive one “PartialOrderFill” message, sent by the exchange, per partial execution noting the fill price of each.

• Order Cancellation The order book locates the requested order by unique order id, removes any remaining unfilled quantity from the order book, and sends an “order cancelled“ message via the exchange.
• Regenerate Queue Depletion: Whenever a queue is depleted or there is a price shift, the Exchange automatically regenerates a queue to be used for the new price level.
• AutoCancellation: The exchange class automatically cancels any remaining limit orders beyond the first 2 levels of the limit order book anytime there is a price shift. It then appropriately notifies agents accordingly.

Message Class:
Messages are the carriers of information through which different entities exchange information with each other. They are implemented as dataclasses in python. Every time a trading agent wants to make a trade, it has to send a LimitOrderMsg, MarketOrderMsg or CancelOrderMsg depending or DoNothing message on what action it chooses to take. Similarly, every time the exchange wishes to notify an agent that an order has been executed, cancelled or a trade has happened, it has to send a message. Whenever a message is sent, it is added to the priority queue in the Kernel where it is processed in order respectively. Any message that contains information about an Order, such as a LimitOrderMsg, MarketOrderMsg, CancelOrderMsg, or OrderExecutedMsg contains a field with the relevant Order object. 


Order Class:
Trading orders are implemented as the Order class for improved generalisation and extendability of the code. Each Order object has the following necessary fields: 

class Order:
    time_placed: int
    order_id: int=field(init=False)
    side: str
    size: int
    symbol: Optional[str]
    agent_id: int #set to -1 for orders randomly generated by the exchange
    filled: Optional[bool] = False
    cancelled: Optional[bool]=False
    fill_time: Optional[int] = None
    cancel_time: Optional[int]=None
    _level: str=None #private method for internal use
Limit Orders, Market Orders and Cancel orders separately inherit from this base class. All simulation entities communicate information about trading orders by passing objects of this class. This allows integration of the RL environment with various different financial models that may come pre-implemented with their own implementations of financial orders. For example, to support integration with the Hawkes Simulation process as outlined in Konark_Jain (\cite paper) along with various other future financial models, the Arrival_Model class contains an orderwrapper and orderunwrapper function that converts the internal language of the Hawkes Simulation process into an order object and back again. In the context of this paper, the GYM Agent class chooses between 12 discrete actions and the choice of doing nothing. The Agent class therefore contains similar wrappers to convert an action into an order object and back again.





