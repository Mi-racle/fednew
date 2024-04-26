from functools import reduce
from logging import WARNING
from typing import Optional, Callable, Dict, Tuple, List, Union

import numpy as np
from flwr.common import NDArrays, Scalar, Parameters, MetricsAggregationFn, log, FitRes, parameters_to_ndarrays, \
    ndarrays_to_parameters, EvaluateRes, FitIns
from flwr.server import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import weighted_loss_avg


class FedNew(FedAvg):

    def __init__(
            self,
            *,
            fraction_fit: float = 1.0,
            fraction_evaluate: float = 1.0,
            min_fit_clients: int = 2,
            min_evaluate_clients: int = 2,
            min_available_clients: int = 2,
            evaluate_fn: Optional[
                Callable[
                    [int, NDArrays, Dict[str, Scalar]],
                    Optional[Tuple[float, Dict[str, Scalar]]],
                ]
            ] = None,
            on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            accept_failures: bool = True,
            initial_parameters: Optional[Parameters] = None,
            fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
            evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
            proximal_mu: float,
    ) -> None:
        super(FedNew, self).__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn
        )
        self.proximal_mu = proximal_mu

    def configure_fit(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training.

        Sends the proximal factor mu to the clients
        """
        # Get the standard client/config pairs from the FedAvg super-class
        client_config_pairs = super().configure_fit(
            server_round, parameters, client_manager
        )

        # Return client/config pairs with the proximal factor mu added
        return [
            (
                client,
                FitIns(
                    fit_ins.parameters,
                    {**fit_ins.config, 'proximal_mu': self.proximal_mu},
                ),
            )
            for client, fit_ins in client_config_pairs
        ]

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.metrics['distribution'])
            for _, fit_res in results
        ]
        aggregated_ndarrays = self._aggregate(weights_results)

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        # centralized
        # fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
        # loss_aggregated = weighted_loss_avg(
        #     [
        #         (num_examples, metrics['loss'])
        #         for num_examples, metrics in fit_metrics
        #     ]
        # )
        # accuracy_aggregated = weighted_loss_avg(
        #     [
        #         (num_examples, metrics['accuracy'])
        #         for num_examples, metrics in fit_metrics
        #     ]
        # )
        # confusion_matrix = [[0 for __ in range(53)] for _ in range(53)]
        # for _, metrics in fit_metrics:
        #     confusion_matrix += metrics['confusion_matrix']
        # confusion_matrix = np.array(confusion_matrix)
        #
        # with open(f'runs/loss_central.txt', 'a+') as fout:
        #     fout.write(str(loss_aggregated) + '\n')
        # with open(f'runs/accuracy_central.txt', 'a+') as fout:
        #     fout.write(str(accuracy_aggregated) + '\n')
        # np.savetxt(f'runs/confusion_matrix_central.csv', confusion_matrix, delimiter=',')

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )
        confusion_matrix = [[0 for __ in range(53)] for _ in range(53)]
        for _, evaluate_res in results:
            confusion_matrix += evaluate_res.metrics['confusion_matrix']
        confusion_matrix = np.array(confusion_matrix)
        np.savetxt(f'runs/confusion_matrix_avg.csv', confusion_matrix, delimiter=',')

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        with open(f'runs/loss_fedavg.txt', 'a+') as fout:
            fout.write(str(loss_aggregated) + '\n')
        with open(f'runs/accuracy_fedavg.txt', 'a+') as fout:
            fout.write(str(metrics_aggregated['accuracy']) + '\n')

        return loss_aggregated, metrics_aggregated

    def _aggregate(self, results: List[Tuple[NDArrays, int]]) -> NDArrays:
        """Compute weighted average."""

        distributions = [distribution for (_, distribution) in results]



        # Create a list of weights, each multiplied by the related number of examples
        # list[list[ndarray]] / [num_clients][num_layers][...]
        weighted_weights = [
            [layer * num_examples for layer in weights] for weights, num_examples in results
        ]

        # Compute average weights of each layer
        weights_prime: NDArrays = [
            reduce(np.add, layer_updates) / num_examples_total
            for layer_updates in zip(*weighted_weights)
        ]

        return weights_prime
