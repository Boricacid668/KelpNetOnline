using System;
using KelpNet.CPU;

namespace KelpNet
{
    /// <summary>
    /// Provides incremental (online) training utilities for <see cref="FunctionStack{T}"/> models.
    /// </summary>
    public class IncrementalTrainer<T, TLabel>
        where T : unmanaged, IComparable<T>
        where TLabel : unmanaged, IComparable<TLabel>
    {
        public FunctionStack<T> Model { get; }
        public Optimizer<T> Optimizer { get; }
        public LossFunction<T, TLabel> LossFunction { get; }

        public IncrementalTrainer(FunctionStack<T> model, Optimizer<T> optimizer, LossFunction<T, TLabel> lossFunction)
        {
            Model = model ?? throw new ArgumentNullException(nameof(model));
            LossFunction = lossFunction ?? throw new ArgumentNullException(nameof(lossFunction));
            Optimizer = optimizer;

            Optimizer?.SetUp(Model);
        }

        public T UpdateIncremental(NdArray<T> input, NdArray<TLabel> target)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (target == null) throw new ArgumentNullException(nameof(target));

            return UpdateIncremental(new[] { input }, new[] { target });
        }

        public T UpdateIncremental(NdArray<T>[] inputs, NdArray<TLabel>[] targets)
        {
            if (inputs == null || inputs.Length == 0)
            {
                throw new ArgumentException("At least one input must be provided.", nameof(inputs));
            }

            if (targets == null || targets.Length == 0)
            {
                throw new ArgumentException("At least one target must be provided.", nameof(targets));
            }

            NdArray<T>[] outputs = Model.Forward(inputs);
            T loss = LossFunction.Evaluate(outputs, targets);

            Model.Backward(outputs);

            if (Optimizer != null)
            {
                Optimizer.Update?.Invoke();
            }

            return loss;
        }

        public NdArray<T>[] Predict(params NdArray<T>[] inputs)
        {
            if (inputs == null || inputs.Length == 0)
            {
                throw new ArgumentException("At least one input must be provided.", nameof(inputs));
            }

            return Model.Predict(inputs);
        }
    }
}
