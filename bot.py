import datasets
from functools import partial
from loguru import logger
from utils import (
    generate_together_stream,
    generate_with_references,
    DEBUG,
)
import typer
from rich import print
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt
from datasets.utils.logging import disable_progress_bar
from time import sleep
from config import Config

disable_progress_bar()

console = Console()

welcome_message = """
¡Bienvenido a la demostración interactiva de Together AI MoA (Mixture-of-Agents)!

Mixture of Agents (MoA) es un enfoque novedoso que aprovecha las fortalezas colectivas de múltiples LLMs (Modelos de Lenguaje Grande) para mejorar el rendimiento, logrando resultados de vanguardia. Al emplear una arquitectura en capas donde cada capa está compuesta por varios agentes LLM, MoA supera significativamente el 57.5% de GPT-4 Omni en AlpacaEval 2.0 con una puntuación de 65.1%, ¡utilizando solo modelos de código abierto!

Este proyecto es un fork realizado por Esteban Calabria de Mixture-of-Agents (MoA) original (https://github.com/togethercomputer/MoA) modificado gracias a la idea de Matthew Berman (https://www.youtube.com/@matthew_berman) para que funcione con Groq.

"""
#CHANGE
default_reference_models = Config.models
#default_reference_models = [
#    "llama3-8b-8192",
#    "llama3-70b-8192",
#    "mixtral-8x7b-32768",
#    "gemma-7b-it"
#]


def process_fn(
    item,
    temperature=0.7,
    max_tokens=2048,
):
    """
    Processes a single item (e.g., a conversational turn) using specified model parameters to generate a response.

    Args:
        item (dict): A dictionary containing details about the conversational turn. It should include:
                     - 'references': a list of reference responses that the model may use for context.
                     - 'model': the identifier of the model to use for generating the response.
                     - 'instruction': the user's input or prompt for which the response is to be generated.
        temperature (float): Controls the randomness and creativity of the generated response. A higher temperature
                             results in more varied outputs. Default is 0.7.
        max_tokens (int): The maximum number of tokens to generate. This restricts the length of the model's response.
                          Default is 2048.

    Returns:
        dict: A dictionary containing the 'output' key with the generated response as its value.
    """

    references = item.get("references", [])
    model = item["model"]
    messages = item["instruction"]

    output = generate_with_references(
        model=model,
        messages=messages,
        references=references,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if DEBUG:
        logger.info(
            f"model: {model}, instruction: {item['instruction']}, output: {output[:20]}"
        )

    print(f"\nFinished querying [bold]{model}.[/bold]")

    return {"output": output}


def main(
    #model: str = "llama3-70b-8192", #CHANGE
    model: str = Config.main_model,
    reference_models: list[str] = default_reference_models,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    rounds: int = 1,
    multi_turn=True,
):
    """
    Runs a continuous conversation between user and MoA.

    Args:
    - model (str): The primary model identifier used for generating the final response. This model aggregates the outputs from the reference models to produce the final response.
    - reference_models (List[str]): A list of model identifiers that are used as references in the initial rounds of generation. These models provide diverse perspectives and are aggregated by the primary model.
    - temperature (float): A parameter controlling the randomness of the response generation. Higher values result in more varied outputs. The default value is 0.7.
    - max_tokens (int): The maximum number of tokens that can be generated in the response. This limits the length of the output from each model per turn. Default is 2048.
    - rounds (int): The number of processing rounds to refine the responses. In each round, the input is processed through the reference models, and their outputs are aggregated. Default is 1.
    - multi_turn (bool): Enables multi-turn interaction, allowing the conversation to build context over multiple exchanges. When True, the system maintains context and builds upon previous interactions. Default is True. When False, the system generates responses independently for each input.
    """
    md = Markdown(welcome_message)
    console.print(md)
    sleep(0.75)
    console.print(
        "\n[bold]To use this demo, answer the questions below to get started [cyan](press enter to use the defaults)[/cyan][/bold]:"
    )

    data = {
        "instruction": [[] for _ in range(len(reference_models))],
        "references": [""] * len(reference_models),
        "model": [m for m in reference_models],
    }

    if not Config.api_key:
        Config.api_key = Prompt.ask("Enter your Groq API key: ")

    num_proc = len(reference_models)

    model = Prompt.ask(
        "\n1. What main mod   el do you want to use?",
        #default="llama3-70b-8192", #CHANGE
        default=Config.main_model,
    )
    console.print(f"Selected {model}.", style="yellow italic")
    temperature = float(
        Prompt.ask(
            "2. What temperature do you want to use? [cyan bold](0.7) [/cyan bold]",
            default=0.7,
            show_default=True,
        )
    )
    console.print(f"Selected {temperature}.", style="yellow italic")
    max_tokens = int(
        Prompt.ask(
            "3. What max tokens do you want to use? [cyan bold](2048) [/cyan bold]",  #CHANGE
            default=2048,  #CHANGE
            show_default=True,
        )
    )
    console.print(f"Selected {max_tokens}.", style="yellow italic")

    while True:

        try:
            instruction = Prompt.ask(
                "\n[cyan bold]Prompt >>[/cyan bold] ",
                default="Top things to do in NYC",
                show_default=True,
            )
        except EOFError:
            break

        if instruction == "exit" or instruction == "quit":
            print("Goodbye!")
            break
        if multi_turn:
            for i in range(len(reference_models)):
                data["instruction"][i].append({"role": "user", "content": instruction})
                data["references"] = [""] * len(reference_models)
        else:
            data = {
                "instruction": [[{"role": "user", "content": instruction}]]
                * len(reference_models),
                "references": [""] * len(reference_models),
                "model": [m for m in reference_models],
            }

        eval_set = datasets.Dataset.from_dict(data)

        with console.status("[bold green]Querying all the models...") as status:
            for i_round in range(rounds):
                eval_set = eval_set.map(
                    partial(
                        process_fn,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    ),
                    batched=False,
                    num_proc=num_proc,
                )
                references = [item["output"] for item in eval_set]
                data["references"] = references
                eval_set = datasets.Dataset.from_dict(data)

        console.print(
            "[cyan bold]Aggregating results & querying the aggregate model...[/cyan bold]"
        )
        output = generate_with_references(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=data["instruction"][0],
            references=references,
            generate_fn=generate_together_stream,
        )

        all_output = ""
        print("\n")
        console.log(Markdown(f"## Final answer from {model}"))

        for chunk in output:
            out = chunk.choices[0].delta.content
            if out is not None:
                console.print(out, end="")
                all_output += out
        print()

        if DEBUG:
            logger.info(
                f"model: {model}, instruction: {data['instruction'][0]}, output: {all_output[:20]}"
            )
        if multi_turn:
            for i in range(len(reference_models)):
                data["instruction"][i].append(
                    {"role": "assistant", "content": all_output}
                )


if __name__ == "__main__":
    typer.run(main)
