// Cost as the API returns it: total cost and cost per success
// (cost_per_success is null when the experiment has no successes).
// Cost per trial is arithmetic on two returned fields (cost / trial_count)
// and is labelled as derived.
import { usd } from "../format";

interface Props {
  cost: number;
  costPerSuccess: number | null;
  trialCount: number;
}

export default function CostTable({ cost, costPerSuccess, trialCount }: Props) {
  return (
    <table className="data-table">
      <tbody>
        <tr>
          <th scope="row">Total cost</th>
          <td>{usd(cost)}</td>
        </tr>
        <tr>
          <th scope="row">Cost per success</th>
          <td>{usd(costPerSuccess, 6)}</td>
        </tr>
        <tr>
          <th scope="row">Cost per trial (derived: cost / trials)</th>
          <td>{trialCount > 0 ? usd(cost / trialCount, 6) : "n/a"}</td>
        </tr>
        <tr>
          <th scope="row">Trials</th>
          <td>{trialCount}</td>
        </tr>
      </tbody>
    </table>
  );
}
