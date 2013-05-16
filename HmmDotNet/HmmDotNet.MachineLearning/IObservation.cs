namespace HmmDotNet.MachineLearning
{
    public interface IObservation<T>
    {
        string Description { get;}
        T Value { get; }
        int Dimention { get; }
    }
}
