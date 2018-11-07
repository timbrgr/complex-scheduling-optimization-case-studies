# Complex Scheduling Case Studies 

## Case Study 1: The Generic Time Scheduling Problem (GTSP)

Given a project plan as directed graph $ G = (V,E) $ with activities $ V $, time lag relations $ E $, are weights $ \delta_{ij}, (i,j) \in E $ and starting times $ S_i, \forall i \in V$.

Formulate the optimization model of the Generic Time Scheduling Problem

![Alt text](img/GTSP.PNG?raw=true "GTSP formulation")

with the objective function to minimize the sum of the (unweighted) starting time (_WST objective function_):

$$ f(S) = \sum_{i \in V} S_i $$

## Case Study 2: Resource-Constrained Project Scheduling Problem (RCPSP) with Pulse Variables

