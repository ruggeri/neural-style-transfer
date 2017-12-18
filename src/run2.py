import generation_network
import loss_network

generation_model = generation_network.build()
print(
    generation_model.output
)
