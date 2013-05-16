namespace HmmDotNet.MachineLearning
{
    public class MultivariateObservation : IObservation<double[]>
    {
        #region Properties

        public string Description { get; protected set; }
        public double[] Value { get; protected set; }
        public int Dimention
        {
            get
            {
                var d = -1;
                if (Value != null)
                {
                    d = Value.Length;
                }
                return d; 
            }
        }

        #endregion Properties

        #region Constructors

        public MultivariateObservation(double[] value, string description)
        {
            Description = description;
            Value = value;
        }

        #endregion Constructors
    }
}
