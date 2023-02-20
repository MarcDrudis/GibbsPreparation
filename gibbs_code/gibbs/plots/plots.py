import numpy as np
from gibbs.dataclass import GibbsResult
from gibbs.learning.klocal_pauli_basis import KLocalPauliBasis


def dataframe(gibbs_result: GibbsResult, timestep: int, periodic: bool = False):
    local_sizes = np.array(
        [0]
        + [
            KLocalPauliBasis(i + 1, gibbs_result.num_qubits, periodic).size
            for i in range(gibbs_result.klocality)
        ]
    )
    local_positions = local_sizes[1:] - local_sizes[:-1]
    local_labels = np.concatenate(
        [[f"{i+1}-local"] * local_positions[i] for i in range(local_positions.size)]
    )
    data = {
        "Pauli Basis Label": KLocalPauliBasis(
            gibbs_result.klocality, gibbs_result.num_qubits, periodic
        ).paulis_list,
        "Pauli Basis": np.arange(
            KLocalPauliBasis(
                gibbs_result.klocality, gibbs_result.num_qubits, periodic
            ).size
        ),
        "locality": local_labels,
        "prepH": np.real(gibbs_result.cfaulties[timestep]),
        "priorH": np.real(gibbs_result.coriginal) * gibbs_result.betas[timestep],
    }
    return data, local_sizes


def axis_locality(fig, k_locality, local_sizes):
    fig.update_layout(
        xaxis=dict(
            tickvals=local_sizes[:-1] - 0.5,
            ticktext=[f"{i+1}-local" for i in range(k_locality)],
            ticks="outside",
            showgrid=True,
            ticklen=5,
        )
    )


def posteriordist(gibbs_result: GibbsResult, posterior_mean, posterior_cov):
    import plotly.express as px
    import plotly.graph_objects as go

    c_original_prior = gibbs_result.coriginal
    data, _ = dataframe(gibbs_result)
    data["posteriorH"]: posterior_mean[: c_original_prior.size]
    data["Posterior Std"]: np.real(posterior_cov.diagonal()[: c_original_prior.size])
    data["Prior Error"] = np.abs(data["priorH"] - data["prepH"])
    data["Posterior Error"] = np.abs(data["posteriorH"] - data["prepH"])

    # fig = px.bar(data, x='paulibasis', y=['prepH','priorH'],barmode="overlay")
    fig = px.bar(
        data,
        x="Pauli Basis",
        y=["Prior Error", "Posterior Error"],
        color_discrete_sequence=["blue", "red"],
        barmode="group",
        hover_name="Pauli Basis Label",
    )
    fig.add_trace(
        go.Scatter(
            x=data["Pauli Basis"] - 0.5,
            y=data["Posterior Std"],
            mode="lines",
            line={"shape": "hv", "color": "black"},
            name="Posterior Std",
            showlegend=True,
        )
    )

    fig.show()


def preparation(
    result: GibbsResult, timestep: int = -1, periodic: bool = False
) -> None:
    import plotly.express as px

    df, local_sizes = dataframe(result, timestep, periodic)
    print(df["prepH"].shape, df["priorH"].shape, df["Pauli Basis"].shape)
    fig = px.bar(
        df,
        x="Pauli Basis",
        y=["prepH", "priorH"],
        barmode="group",
        hover_name="Pauli Basis Label",
    )
    axis_locality(fig, result.klocality, local_sizes)
    fig.update_layout(
        title="Preparation Hamiltonian",
        xaxis_title="Pauli Basis",
        yaxis_title="Coefficient",
        legend_title="",
        font={"size": 18, "color": "black"},
        # legend={"yanchor":"top","y":0.99,"xanchor":"right","x":0.99}
    )
    return fig


def preparation_vsclassic(result: GibbsResult, result_classic: GibbsResult) -> None:
    import plotly.express as px

    df, local_sizes = dataframe(result)
    df["classicprepH"] = np.real(result_classic.cfaulties[-1])
    fig = px.bar(
        df,
        x="Pauli Basis",
        y=["prepH", "priorH", "classicprepH"],
        barmode="group",
        hover_name="Pauli Basis Label",
    )
    axis_locality(fig, result.klocality, local_sizes)
    fig.update_layout(
        title="Preparation Hamiltonian",
        xaxis_title="Pauli Basis",
        yaxis_title="Coefficient",
        legend_title="",
        font={"size": 18, "color": "black"},
        legend={"yanchor": "bottom", "y": 0.01, "xanchor": "right", "x": 0.99},
    )
    return fig


def compare_preparations(results: list[GibbsResult], titles: list[str], timestep: int):
    """Takes a list of results and labels to plot them. Takes the original hamiltonian of the first state."""
    import plotly.express as px

    df, local_sizes = dataframe(results[0], timestep)
    for title, result in zip(titles, results):
        df[title] = np.real(result.cfaulties[timestep])
    fig = px.bar(
        df,
        x="Pauli Basis",
        y=["priorH"] + titles,
        barmode="group",
        hover_name="Pauli Basis Label",
    )
    axis_locality(fig, results[0].klocality, local_sizes)
    fig.update_layout(
        title="Preparation Hamiltonian",
        xaxis_title="Pauli Basis",
        yaxis_title="Coefficient",
        legend_title="",
        font={"size": 18, "color": "black"},
        # legend={"yanchor":"bottom","y":0.01,"xanchor":"right","x":0.99}
    )
    return fig


def preparation_error(result: GibbsResult) -> None:
    import plotly.express as px

    df, local_sizes = dataframe(result)
    df["Preparation Error"] = np.abs(df["prepH"] - df["priorH"])
    fig = px.bar(
        df,
        x="Pauli Basis",
        y="Preparation Error",
        barmode="group",
        hover_name="Pauli Basis Label",
    )
    axis_locality(fig, result.klocality, local_sizes)
    fig.update_layout(
        title="Preparation Error",
        xaxis_title="Pauli Basis",
        legend_title="",
        font={"size": 18, "color": "black"},
        legend={"yanchor": "top", "y": 0.99, "xanchor": "right", "x": 0.99},
    )
    return fig
